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

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "metrics/paths/DeltaSteppingVariants/Buckets.h"
#include "metrics/paths/DeltaSteppingVariants/EpochCommon.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/patterns/comm_patterns_2/QuestionExchange.h>
#include <mpi-wrapper/reductions/MPIReductions.h>
#include <mpi-wrapper/rma/RMAWindow.h>

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Implements the delta-stepping epoch strategy that postpones updates to nodes that would
 *		fall outside of the current bucket based on long or outer-short arcs to the last phase.
 */
class PushInnerShortVariant {
public:
    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Postpones updates of the distance to nodes that would fall outside of the current bucket
     *			based on long or outer-short arcs to the last phase.
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
            using question_list_type = std::vector<mpiPP::comm_patterns_2::TargetedQuestion<question_data_type, node_id_type>>;

            auto long_arcs = std::unordered_map<node_id_type, std::vector<OutArc>>{};
            long_arcs.reserve(number_local_nodes);

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

            // Collects the questions of every local node, exchanges them, and answers the received ones
            const auto ask_questions_and_answer = [number_local_nodes](const auto& generate_questions, const auto& generate_answer) {
                auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<question_data_type, node_id_type>(number_local_nodes);
                for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
                    question_builder.add_questions(node_id, generate_questions(node_id));
                }

                const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());
                for (const auto& question : questions_to_answer) {
                    generate_answer(question.target_node, question.parameter);
                }
            };

            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                auto question_function = [&active_nodes, &distances, &long_arcs, delta, current_epoch, my_rank, my_rank_cast,
                                          &graph](const node_id_type node_id) {
                    if (!active_nodes.contains(node_id)) {
                        return question_list_type{};
                    }

                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = question_list_type{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        const auto new_distance = my_distance + utility::safe_cast<distance_type>(weight);

                        auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));
                        if (old_distance <= new_distance) {
                            continue;
                        }

                        const auto new_bucket = EpochCommon::compute_bucket_index(delta, new_distance);
                        if (new_bucket > current_epoch) {
                            // This is a long arc or an outer-short arc
                            long_arcs[node_id].emplace_back(target_rank, target_node_id, weight);
                            continue;
                        }

                        if (new_distance < old_distance) {
                            return_value.emplace_back(target_rank, target_node_id, new_distance);
                        }
                    }

                    return return_value;
                };

                ask_questions_and_answer(question_function, answer_function);
            }

            auto question_function_long_arcs = [&distances, &long_arcs, my_rank](const node_id_type node_id) {
                const auto long_arcs_it = long_arcs.find(node_id);
                if (long_arcs_it == long_arcs.end()) {
                    return question_list_type{};
                }

                const auto& out_arcs = long_arcs_it->second;

                auto return_value = question_list_type{};
                return_value.reserve(out_arcs.size());

                for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                    const auto my_distance = distances.get(node_id, my_rank);

                    const auto new_distance = my_distance + utility::safe_cast<distance_type>(weight);

                    auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));

                    if (new_distance < old_distance) {
                        return_value.emplace_back(target_rank, target_node_id, new_distance);
                    }
                }

                return return_value;
            };

            ask_questions_and_answer(question_function_long_arcs, answer_function);
        };

        for (auto current_epoch = std::size_t{ 0 }; EpochCommon::check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }
};
