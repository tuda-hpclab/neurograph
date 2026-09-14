#pragma once

/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/paths/DeltaSteppingVariants/Buckets.h"
#include "metrics/paths/DeltaSteppingVariants/EpochCommon.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/patterns/comm_patterns_2/AnswerExchange.h>
#include <mpi-wrapper/patterns/comm_patterns_2/QuestionExchange.h>
#include <mpi-wrapper/reductions/MPIReductions.h>
#include <mpi-wrapper/rma/RMAWindow.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * @brief Implements the delta-stepping epoch strategy that postpones the processing of long arcs
 *		(arcs with weight >= delta) to the last phase, and pulls in that last phase rather than pushes.
 */
class PullModelVariant {
    constexpr static auto max_distance = std::numeric_limits<distance_type>::max();

public:
    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Postpones the processing of long arcs (arcs with weight >= delta) to the last phase.
     *		- Pulls in the last phase rather than pushes.
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
                               mpiPP::RMAWindow<distance_type>& distances) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();

        const auto number_local_nodes = graph.get_number_local_nodes();

        auto buckets = Buckets(number_local_nodes);
        if (my_rank_cast == root_node.owning_mpi_rank) {
            buckets.move_node(root_node.node_id, 0);
            buckets.mark_as_active(root_node.node_id);
        }

        const auto process_bucket = [&graph, &distances, &buckets, delta, my_rank, my_rank_cast, number_local_nodes](const std::size_t current_epoch) {
            using question_data_type = distance_type;
            using pull_question_data_type = std::tuple<mpi_rank_type, node_id_type, distance_type>;
            using pull_answer_type = distance_type;

            using question_list_type = std::vector<mpiPP::comm_patterns_2::TargetedQuestion<question_data_type, node_id_type>>;
            using pull_question_list_type = std::vector<mpiPP::comm_patterns_2::TargetedQuestion<pull_question_data_type, node_id_type>>;

            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                auto question_function = [&distances, delta, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = question_list_type{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        if (utility::safe_cast<distance_type>(weight) >= delta) {
                            // This is a long arc and will be processed by pulling
                            continue;
                        }

                        const auto new_distance = my_distance + utility::safe_cast<distance_type>(weight);
                        return_value.emplace_back(target_rank, target_node_id, new_distance);
                    }

                    return return_value;
                };

                auto answer_function = [&distances, &buckets, my_rank, delta](const node_id_type node_id, const question_data_type new_distance) {
                    const auto old_distance = distances.get(node_id, my_rank);
                    if (old_distance <= new_distance) {
                        return;
                    }

                    distances.put(new_distance, node_id, my_rank);

                    buckets.mark_as_active(node_id);

                    const auto old_bucket_id = old_distance / delta;
                    const auto new_bucket_id = new_distance / delta;

                    if (new_bucket_id < old_bucket_id) {
                        buckets.move_node(node_id, new_bucket_id);
                    }
                };

                auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<question_data_type, node_id_type>(number_local_nodes);
                for (const auto node_id : active_nodes) {
                    question_builder.add_questions(node_id, question_function(node_id));
                }

                const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());
                for (const auto& question : questions_to_answer) {
                    answer_function(question.target_node, question.parameter);
                }
            }

            auto question_function_pull = [&distances, &buckets, delta, current_epoch, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                const auto bucket_index = buckets.get_current_bucket(node_id);
                if (bucket_index <= current_epoch) {
                    return pull_question_list_type{};
                }

                const auto my_distance = distances.get(node_id, my_rank);
                const auto& in_arcs = graph.get_in_arcs(my_rank_cast, node_id);

                auto return_value = pull_question_list_type{};
                return_value.reserve(in_arcs.size());

                for (const auto& [source_rank, source_node_id, weight] : in_arcs) {
                    if (utility::safe_cast<distance_type>(weight) >= my_distance - current_epoch * delta) {
                        continue;
                    }

                    return_value.emplace_back(source_rank, source_node_id, std::tuple(my_rank.get_rank(), node_id, my_distance));
                }

                return return_value;
            };

            auto answer_function_pull = [&distances, my_rank, my_rank_cast, &graph](const node_id_type node_id,
                                                                                    const pull_question_data_type questioner) -> pull_answer_type {
                const auto& [question_rank, question_node_id, question_distance] = questioner;

                const auto my_distance = distances.get(node_id, my_rank);
                if (my_distance >= question_distance) {
                    return max_distance;
                }

                const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);
                for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                    if (target_rank != question_rank || target_node_id != question_node_id) {
                        continue;
                    }

                    // We found the arc to the questioning node
                    if (my_distance + utility::safe_cast<distance_type>(weight) < question_distance) {
                        return my_distance + utility::safe_cast<distance_type>(weight);
                    }

                    break;
                }

                return max_distance;
            };

            auto pull_question_builder = mpiPP::comm_patterns_2::QuestionBuilder<pull_question_data_type, node_id_type>(number_local_nodes);
            for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
                pull_question_builder.add_questions(node_id, question_function_pull(node_id));
            }

            auto pull_questions_asked = pull_question_builder.finalize();
            const auto pull_questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(pull_questions_asked);

            auto pull_answer_builder = mpiPP::comm_patterns_2::AnswerBuilder<pull_answer_type>(pull_questions_to_answer);
            for (const auto& question : pull_questions_to_answer) {
                pull_answer_builder.push_answer(answer_function_pull(question.target_node, question.parameter));
            }

            const auto pull_answers = mpiPP::comm_patterns_2::exchange_answers(pull_answer_builder.finalize(), std::move(pull_questions_asked));

            for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
                const auto& answers_to_me = pull_answers.get_answers_of_questioner_node(node_id);
                if (answers_to_me.empty()) {
                    continue;
                }

                const auto my_distance = distances.get(node_id, my_rank);
                const auto new_distance = *std::ranges::min_element(answers_to_me);
                if (new_distance >= my_distance) {
                    continue;
                }

                distances.put(new_distance, node_id, my_rank);

                const auto new_bucket_id = new_distance / delta;
                buckets.move_node(node_id, new_bucket_id);
            }
        };

        for (auto current_epoch = std::size_t{ 0 }; EpochCommon::check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }
};
