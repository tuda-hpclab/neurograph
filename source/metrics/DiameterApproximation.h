#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/AllPairsShortestPath.h"
#include "utility/Timer.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"
#include "mpi-wrapper/RMAWindow.h"
#include "mpi-wrapper/comm_patterns/Questions.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>

class DiameterApproximation {
    using iteration_type = std::size_t;

    constexpr static auto max_distance = std::numeric_limits<distance_type>::max();
    constexpr static auto max_generation = std::numeric_limits<iteration_type>::max();

public:
    [[nodiscard]] static distance_type compute_approximation(const DistributedGraph& graph, const distance_type r) {
        auto timer = Timer{ "Diameter Approximation" };

        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto number_global_nodes = NodeCounter::all_count_nodes(graph);

        const auto last_iteration = static_cast<iteration_type>(std::ceil(std::log2(number_global_nodes)));

        auto rma_windows = construct_rma_windows(number_local_nodes);

        timer.time_section("Construct RMA windows");

        for (auto iteration = iteration_type(0); iteration < last_iteration; iteration++) {
            const auto has_center_locally = mark_as_centers(number_local_nodes, utility::save_cast<node_id_type>(number_global_nodes), iteration, rma_windows);
            const auto has_center_globally = mpiPP::MPIReductions::all_reduce_or(has_center_locally);
            if (!has_center_globally) {
                continue;
            }

            perform_growing_step(graph, r, iteration, rma_windows);

            stabilize_nodes(number_local_nodes, rma_windows);

            const auto all_locally_stable = check_all_stable(rma_windows);
            const auto all_globally_stable = mpiPP::MPIReductions::all_reduce_and(all_locally_stable);
            if (all_globally_stable) {
                break;
            }
        }

        timer.time_section("Construct Cluster");

        const auto contracted_graph = contract(graph, rma_windows);

        timer.time_section("Contruct Graph");

        timer.time_section("Delete RMA windows");

        const auto [_1, _2, approximated_diameter, _3, _4] = AllPairsShortestPath::compute_apsp(contracted_graph);

        timer.time_section("Small APSP");
        timer.finish();

        return approximated_diameter + 2 * r;
    }

private:
    struct RMAWindowHolder {
        mpiPP::RMAWindow<NodeIdentifier> centers;
        mpiPP::RMAWindow<distance_type> distances;
        mpiPP::RMAWindow<iteration_type> generation;
        mpiPP::RMAWindow<std::uint8_t> stable;
    };

    [[nodiscard]] static RMAWindowHolder construct_rma_windows(const node_id_type number_local_nodes) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        auto rma_windows = RMAWindowHolder{ .centers = mpiPP::RMAWindow<NodeIdentifier>(number_local_nodes),
                                            .distances = mpiPP::RMAWindow<distance_type>(number_local_nodes),
                                            .generation = mpiPP::RMAWindow<iteration_type>(number_local_nodes),
                                            .stable = mpiPP::RMAWindow<std::uint8_t>(number_local_nodes) };

        for (auto i = 0U; i < number_local_nodes; i++) {
            rma_windows.centers.put(NodeIdentifier::uninitialized(), i, my_rank);
            rma_windows.distances.put(max_distance, i, my_rank);
            rma_windows.generation.put(max_generation, i, my_rank);
            rma_windows.stable.put(std::uint8_t{ 0 }, i, my_rank);
        }

        return rma_windows;
    }

    [[nodiscard]] static bool mark_as_centers(const node_id_type number_local_nodes, const node_id_type number_global_nodes, const iteration_type iteration,
                                              RMAWindowHolder& rma_windows) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();

        auto prng = std::mt19937{};
        prng.seed(utility::save_cast<unsigned int>(my_rank_cast));
        auto distr = std::uniform_real_distribution<double>(0.0, 1.0);

        const auto threshold = std::pow(2.0, iteration) / static_cast<double>(number_global_nodes);

        auto has_actual_center = false;

        for (auto node_id = node_id_type(0); node_id < number_local_nodes; node_id++) {
            if (rma_windows.centers.get(node_id, my_rank) != NodeIdentifier::uninitialized()) {
                has_actual_center = true;
                continue;
            }

            const auto random_number = distr(prng);
            if (random_number >= threshold) {
                continue;
            }

            mark_as_center(node_id, iteration, rma_windows);

            has_actual_center = true;
        }

        return has_actual_center;
    }

    static void mark_as_center(const node_id_type node_id, const iteration_type iteration, RMAWindowHolder& rma_windows) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();

        rma_windows.centers.put(NodeIdentifier{ my_rank_cast, node_id }, node_id, my_rank);
        rma_windows.distances.put(distance_type(0), node_id, my_rank);
        rma_windows.generation.put(iteration, node_id, my_rank);
        rma_windows.stable.put(std::uint8_t(1), node_id, my_rank);
    }

    [[nodiscard]] static bool check_all_stable(RMAWindowHolder& rma_windows) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto number_local_nodes = rma_windows.stable.get_local_size();

        for (auto node_id = node_id_type(0); node_id < number_local_nodes; node_id++) {
            if (rma_windows.stable.get(node_id, my_rank) == std::uint8_t{ 0 }) {
                return false;
            }
        }

        return true;
    }

    static void perform_growing_step(const DistributedGraph& graph, const distance_type r, const iteration_type iteration, RMAWindowHolder& rma_windows) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();

        using question_type = std::tuple<NodeIdentifier, distance_type, iteration_type>;

        auto something_changed_locally = false;

        const auto generate_question = [my_rank, my_rank_cast, r, iteration, &rma_windows, &graph](const node_id_type node_id) {
            using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_type, node_id_type>::result_type;

            const auto my_center = rma_windows.centers.get(node_id, my_rank);
            if (my_center == NodeIdentifier::uninitialized()) {
                return ResultType{};
            }

            const auto out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

            auto targets = ResultType{};
            targets.reserve(out_arcs.size());

            const auto my_distance = rma_windows.distances.get(node_id, my_rank);
            const auto my_generation = rma_windows.generation.get(node_id, my_rank);

            for (const auto& out_arc : out_arcs) {
                const auto& [target_rank, target_node_id, weight] = out_arc;

                if (utility::save_cast<distance_type>(weight) > 2 * r) {
                    continue;
                }

                const auto target_stable = rma_windows.stable.get(target_node_id, mpiPP::MPIRank(target_rank));
                if (target_stable == std::uint8_t(1)) {
                    continue;
                }

                const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);
                const auto threshold = (iteration - my_generation + 1) * 2 * r;
                if (threshold < new_distance) {
                    continue;
                }

                const auto other_distance = rma_windows.distances.get(target_node_id, mpiPP::MPIRank(target_rank));
                if (new_distance >= other_distance) {
                    continue;
                }

                const auto value = std::make_tuple(my_center, new_distance, my_generation);
                targets.emplace_back(target_rank, target_node_id, value);
            }

            return targets;
        };

        const auto generate_answers = [my_rank, &something_changed_locally, &rma_windows](const node_id_type node_id, const question_type question) {
            const auto& [new_center, new_distance, new_generation] = question;

            const auto old_distance = rma_windows.distances.get(node_id, my_rank);
            if (old_distance <= new_distance) {
                return;
            }

            rma_windows.centers.put(new_center, node_id, my_rank);
            rma_windows.distances.put(new_distance, node_id, my_rank);
            rma_windows.generation.put(new_generation, node_id, my_rank);
            rma_windows.stable.put(std::uint8_t(0), node_id, my_rank);

            something_changed_locally = true;
        };

        while (true) {
            something_changed_locally = false;

            mpiPP::comm_patterns::node_to_node_question<question_type, node_id_type>(graph.get_number_local_nodes(), generate_question, generate_answers);

            const auto something_changed_globally = mpiPP::MPIReductions::all_reduce_or(something_changed_locally);
            if (!something_changed_globally) {
                break;
            }
        }
    }

    static void stabilize_nodes(const node_id_type number_local_nodes, RMAWindowHolder& rma_windows) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        for (auto node_id = node_id_type(0); node_id < number_local_nodes; node_id++) {
            const auto center = rma_windows.centers.get(node_id, my_rank);
            if (center != NodeIdentifier::uninitialized()) {
                rma_windows.stable.put(std::uint8_t(1), node_id, my_rank);
            }
        }
    }

    [[nodiscard]] static DistributedGraph contract(const DistributedGraph& graph, RMAWindowHolder& rma_windows) {
        struct OutArcHolder {
            OutArc oa;
            NodeIdentifier new_source_center;
            NodeIdentifier new_target_center;
        };

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto to_correct_locally = std::unordered_set<node_id_type>{};
        to_correct_locally.reserve(number_local_nodes);

        auto to_correct_globally = std::unordered_set<node_id_type>{};
        to_correct_globally.reserve(number_local_nodes);

        const auto get_node_translation = [number_local_nodes, my_rank, my_rank_cast, &to_correct_locally, &to_correct_globally, &rma_windows]() {
            // Counts the number of nodes in the new graph,
            // i.e., #(centers)
            auto number_required_nodes = 0;

            // Maps from local node id to center in new graph
            auto node_translation = std::vector<NodeIdentifier>{};
            node_translation.reserve(number_local_nodes);

            // This will be the inverse of the local portion of node_translation
            auto anti_node_translation = std::unordered_map<node_id_type, node_id_type>{};
            anti_node_translation.reserve(number_local_nodes);

            for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
                const auto& center = rma_windows.centers.get(node_id, my_rank);
                utility::Exception::check(center != NodeIdentifier::uninitialized(), "Some nodes was undiscovered");

                const auto& [center_rank, center_node_id] = center;
                if (center_rank == my_rank_cast && center_node_id == node_id) {
                    // Node is the own center
                    node_translation.emplace_back(my_rank.get_rank(), number_required_nodes);
                    anti_node_translation[utility::save_cast<node_id_type>(number_required_nodes)] = node_id;
                    number_required_nodes++;
                    continue;
                }

                // The node was found by a different center
                node_translation.emplace_back(center_rank, center_node_id);

                if (center_rank == my_rank_cast) {
                    to_correct_locally.emplace(node_id);
                } else {
                    to_correct_globally.emplace(node_id);
                }
            }

            return std::tuple{ node_translation, anti_node_translation };
        };
        auto [node_translation, anti_node_translation] = get_node_translation();

        const auto correct_local_translation = [&to_correct_locally, &node_translation]() {
            for (const auto node_id : to_correct_locally) {
                // The center might have shifted its id
                const auto& [_, pre_contraction_center_node_id] = node_translation[node_id];
                node_translation[node_id] = node_translation[pre_contraction_center_node_id];
            }
        };
        correct_local_translation();

        const auto correct_global_translation = [to_correct_globally, &node_translation, &graph]() {
            using question_type_node_translation = node_id_type;
            using answer_type_node_translation = NodeIdentifier;

            const auto generate_questions_node_translation = [&node_translation](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_type_node_translation, node_id_type>::result_type;

                auto targets = ResultType{};

                const auto& [center_rank, center_node_id] = node_translation[node_id];
                // Actually, we don't need to send something at all
                targets.emplace_back(center_rank, center_node_id, center_node_id);

                return targets;
            };

            const auto generate_answers_node_translation = [&node_translation](const node_id_type node_id, const question_type_node_translation /*question*/) {
                // That is the new center/to where node_id is contracted;
                // it might shift MPI ranks
                return node_translation[node_id];
            };

            using id_range_type = decltype(to_correct_globally);
            const auto& answers_node_translation = mpiPP::comm_patterns::node_to_node_question<question_type_node_translation, answer_type_node_translation, id_range_type, node_id_type>(
                graph.get_number_local_nodes(), to_correct_globally, generate_questions_node_translation, generate_answers_node_translation);

            for (const auto node_id : to_correct_globally) {
                const auto& answers_for_node = answers_node_translation.get_answers_of_questioner_node(node_id);
                // We asked exactly one question
                const auto& new_center_node_id = answers_for_node[0];
                node_translation[node_id] = new_center_node_id;
            }
        };
        correct_global_translation();

        const auto number_required_nodes = anti_node_translation.size();

        /* Now, node_translation correctly points to the ids in the new graph */

        const auto produce_out_arcs = [number_required_nodes, number_local_nodes, my_rank, my_rank_cast, &node_translation, &rma_windows, &graph]() {
            const auto generate_questions_arc_translation = [my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<node_id_type, node_id_type>::result_type;

                const auto out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                auto targets = ResultType{};
                targets.reserve(out_arcs.size());

                for (const auto& [target_rank, target_node_id, _] : out_arcs) {
                    targets.emplace_back(target_rank, target_node_id, node_id);
                }

                return targets;
            };

            const auto generate_answers_arc_translation = [&node_translation, &rma_windows](const node_id_type node_id, const node_id_type /*question*/) {
                // This is where the new target is
                const auto& contracted_center = node_translation[node_id];
                return contracted_center;
            };

            const auto& answers_arc_translation = mpiPP::comm_patterns::node_to_node_question<node_id_type, NodeIdentifier, node_id_type>(
                graph.get_number_local_nodes(), generate_questions_arc_translation, generate_answers_arc_translation);

            auto out_arcs_to_fix = std::unordered_map<node_id_type, std::vector<OutArcHolder>>{};
            out_arcs_to_fix.reserve(number_local_nodes);

            auto new_out_arcs = LoadedArcs{};
            new_out_arcs.resize(number_required_nodes);

            // We need to bend the arcs; some might even move start and end point
            for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
                const auto& answers_for_node = answers_arc_translation.get_answers_of_questioner_node(node_id);
                const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                const auto& new_center_source = node_translation[node_id];
                const auto& [new_center_rank, new_center_node_id] = new_center_source;

                for (auto arc_id = std::size_t{ 0 }; arc_id < out_arcs.size(); arc_id++) {
                    const auto& out_arc = out_arcs[arc_id];
                    const auto& [old_target_rank, old_target_node_id, weight] = out_arc;

                    const auto& new_center_target = answers_for_node[arc_id];
                    const auto& [new_target_rank, new_target_node_id] = new_center_target;

                    const auto source_distance = rma_windows.distances.get(node_id, my_rank);
                    const auto target_distance = rma_windows.distances.get(old_target_node_id, mpiPP::MPIRank(old_target_rank));

                    const auto new_weight = utility::save_cast<distance_type>(weight) + source_distance + target_distance;

                    if (new_center_rank == my_rank_cast) {
                        // The source did not move (new_target_rank == old_target_rank possible)
                        new_out_arcs[new_center_node_id][NodeIdentifier{ new_target_rank, new_target_node_id }] = utility::save_cast<weight_type>(new_weight);
                    } else {
                        // Source moved (new_target_rank == old_target_rank possible)
                        auto new_out_arc = out_arc;
                        new_out_arc.weight = utility::save_cast<weight_type>(new_weight);
                        out_arcs_to_fix[node_id].emplace_back(new_out_arc, new_center_source, new_center_target);
                    }
                }
            }

            const auto generate_questions_move_arcs = [&out_arcs_to_fix](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<OutArc, node_id_type>::result_type;
                const auto& to_fix = out_arcs_to_fix[node_id];

                auto targets = ResultType{};
                targets.reserve(to_fix.size());

                for (const auto& [out_arc, new_source_center, new_target_center] : to_fix) {
                    const auto& [new_source_rank, new_source_node_id] = new_source_center;
                    const auto& [new_target_rank, new_target_node_id] = new_target_center;
                    targets.emplace_back(new_source_rank, new_source_node_id, OutArc{ new_target_rank, new_target_node_id, out_arc.weight });
                }

                return targets;
            };

            const auto generate_answers_move_arcs = [&new_out_arcs](const node_id_type node_id, const OutArc& out_arc) {
                new_out_arcs[node_id][{ out_arc.target_rank, out_arc.target_id }] = out_arc.weight;
            };

            mpiPP::comm_patterns::node_to_node_question<OutArc, node_id_type>(graph.get_number_local_nodes(), generate_questions_move_arcs, generate_answers_move_arcs);

            return new_out_arcs;
        };
        const auto new_out_arcs = produce_out_arcs();

        /* Now, new_out_arcs contains all the arcs that start at each rank */

        const auto produce_in_arcs = [number_required_nodes, my_rank, my_rank_cast, &new_out_arcs, &graph]() {
            auto new_in_arcs = LoadedArcs{};
            new_in_arcs.resize(number_required_nodes);

            const auto generate_questions_mirror_arcs = [&new_out_arcs, my_rank, my_rank_cast, number_required_nodes](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<InArc, node_id_type>::result_type;

                if (node_id >= number_required_nodes) {
                    return ResultType{};
                }

                const auto& out_arcs = new_out_arcs[node_id];

                auto targets = ResultType{};
                targets.reserve(out_arcs.size());

                for (const auto& [target, weight] : out_arcs) {
                    const auto& [target_rank, target_node_id] = target;
                    targets.emplace_back(target_rank, target_node_id, InArc{ my_rank_cast, node_id, weight });
                }

                return targets;
            };

            const auto generate_answers_mirror_arcs = [&new_in_arcs](const node_id_type node_id, const InArc& in_arc) {
                const auto& [source_rank, source_node_id, weight] = in_arc;
                new_in_arcs[node_id].emplace(std::pair{ source_rank, source_node_id }, weight);
            };

            mpiPP::comm_patterns::node_to_node_question<InArc, node_id_type>(graph.get_number_local_nodes(), generate_questions_mirror_arcs, generate_answers_mirror_arcs);

            return new_in_arcs;
        };
        const auto new_in_arcs = produce_in_arcs();

        /* Now, new_in_arcs contains all mirrored out arcs from somewhere else */

        const auto produce_new_nodes = [&graph, &anti_node_translation, number_required_nodes, my_rank, my_rank_cast]() {
            auto new_nodes = LoadedNodes{};

            const auto old_area_names = graph.get_local_area_names();
            new_nodes.area_names.assign(old_area_names.begin(), old_area_names.end());

            const auto old_signal_types = graph.get_local_signal_types();
            new_nodes.signal_types.assign(old_signal_types.begin(), old_signal_types.end());

            new_nodes.positions.resize(number_required_nodes);
            new_nodes.area_names_ind.resize(number_required_nodes);
            new_nodes.signal_types_ind.resize(number_required_nodes);

            for (auto new_node_id = node_id_type{ 0 }; new_node_id < number_required_nodes; new_node_id++) {
                const auto old_node_id = anti_node_translation[new_node_id];

                new_nodes.positions[new_node_id] = graph.get_node_position(my_rank_cast, old_node_id);
                new_nodes.area_names_ind[new_node_id] = graph.get_node_area_localID(my_rank_cast, old_node_id);
                new_nodes.signal_types_ind[new_node_id] = graph.get_node_signal_localID(my_rank_cast, old_node_id);
            }

            return new_nodes;
        };

        return DistributedGraph::construct_graph(produce_new_nodes(), new_out_arcs, new_in_arcs);
    }
};
