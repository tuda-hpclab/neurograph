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

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "metrics/Distance.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"
#include "mpi-wrapper/MPISynchronization.h"
#include "mpi-wrapper/RMAWindow.h"
#include "mpi-wrapper/comm_patterns/Questions.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * Implements the buckets used in delta stepping.
 * Holds a set of active nodes and list of buckets, each holding a set of nodes.
 */
class Buckets {
public:
    constexpr static std::size_t infinity_bucket_index = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Constructs the buckets with the given number of local nodes.
     * @param number_local_nodes The number of local nodes
     */
    explicit Buckets(const node_id_type number_local_nodes) {
        node_to_bucket.resize(utility::save_cast<std::size_t>(number_local_nodes), infinity_bucket_index);
        infinity_bucket.reserve(utility::save_cast<std::size_t>(number_local_nodes) * 2);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
            infinity_bucket.emplace(node_id);
        }
    }

    /**
     * @brief Returns the currently active nodes and clears them
     * @return The currently active nodes
     */
    [[nodiscard]] std::unordered_set<node_id_type> get_active_nodes() noexcept {
        auto temp_value = std::move(active_nodes);
        active_nodes.clear();
        return temp_value;
    }

    /**
     * @brief Moves a specified node to the new bucket. Does not mark the node as active.
     * @param node_id The local node
     * @param bucket_index The new bucket for the nodes
     */
    void move_node(const node_id_type node_id, const std::size_t bucket_index) {
        const auto previous_bucket_index = node_to_bucket[node_id];
        node_to_bucket[node_id] = bucket_index;

        if (previous_bucket_index == infinity_bucket_index) {
            infinity_bucket.erase(node_id);
        } else {
            buckets[previous_bucket_index].erase(node_id);
        }

        if (bucket_index >= buckets.size()) {
            buckets.resize(bucket_index * 2 + 1);
        }

        buckets[bucket_index].emplace(node_id);
    }

    /**
     * @brief Marks the node as active
     * @param node_id The local node id
     */
    void mark_as_active(const node_id_type node_id) {
        active_nodes.emplace(node_id);
    }

    /**
     * @brief Checks if there is an unprocessed node from the current bucket upwards.
     *		Disregards all nodes that are still 'infinitely far away'.
     * @param current_bucket The bucket from where to check
     * @return True iff there are nodes left on this rank
     */
    [[nodiscard]] bool has_nodes_left(const std::size_t current_bucket) const noexcept {
        if (current_bucket >= buckets.size()) {
            // There are no nodes left in this bucket and the larger ones, so this rank is finished.
            // We don't check infinity_bucket because of potentially unreachable nodes.
            return false;
        }

        for (auto bucket_index = current_bucket; bucket_index < buckets.size(); bucket_index++) {
            if (!buckets[bucket_index].empty()) {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Returns a view of all buckets starting at a given index
     * @param current_bucket Where to start
     * @return The view
     */
    [[nodiscard]] std::span<const std::unordered_set<node_id_type>> get_buckets(const std::size_t current_bucket) const noexcept {
        if (buckets.size() <= current_bucket) {
            return {};
        }

        return { buckets.data() + current_bucket, buckets.size() - current_bucket };
    }

    /**
     * @brief Returns a constant reference to the infinity bucket (which might be empty)
     * @return All nodes that are still 'infinitely far away' from the root of the SSSP
     */
    [[nodiscard]] const std::unordered_set<node_id_type>& get_nodes_at_infinity() const noexcept {
        return infinity_bucket;
    }

    /**
     * @brief Returns the current bucket the local node is in
     * @param node_id The id of the node
     * @return The index of the bucket. Can be infinity_bucket_index to represent the infinity bucket.
     */
    [[nodiscard]] std::size_t get_current_bucket(const node_id_type node_id) const noexcept {
        return node_to_bucket[node_id];
    }

private:
    std::vector<std::size_t> node_to_bucket{};

    std::vector<std::unordered_set<node_id_type>> buckets{};
    std::unordered_set<node_id_type> infinity_bucket{};

    std::unordered_set<node_id_type> active_nodes{};
};

class DeltaStepping {
    constexpr static auto max_distance = std::numeric_limits<distance_type>::max();

public:
    [[nodiscard]] static sssp_result<distance_type> compute_shortest_distances(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
                                                                               const DeltaSteppingEpochType epoch_type) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto distance_window = mpiPP::RMAWindow<distance_type>(number_local_nodes);

        for (auto i = 0U; i < number_local_nodes; i++) {
            distance_window.put(max_distance, i, my_rank);
        }

        if (root_node.owning_mpi_rank == my_rank_cast) {
            distance_window.put(0, root_node.node_id, my_rank);
        }

        switch (epoch_type) {
        case DeltaSteppingEpochType::PushBruteForce:
            compute_epochs_push_brute_force(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushCheck:
            compute_epochs_push_previous_check(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushShortLong:
            compute_epochs_push_short_long(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushInnerShort:
            compute_epochs_push_inner_short(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PullModel:
            compute_epochs_pull_model(graph, root_node, delta, distance_window);
            break;
        default:
            utility::Exception::fail("Epoch type is unkown");
        }

        const auto span = std::span{ distance_window.get_pointer(), distance_window.get_local_size() };

        const auto [local_sum_distances, local_max_distance, local_sum_efficiency, local_number_unreached] = summarize_distances(span);

        const auto global_sum_distances = mpiPP::MPIReductions::reduce_sum(local_sum_distances);
        const auto global_max_distance = mpiPP::MPIReductions::reduce_max(local_max_distance);
        const auto global_sum_efficiency = mpiPP::MPIReductions::reduce_sum(local_sum_efficiency);
        const auto global_number_unreached = mpiPP::MPIReductions::reduce_sum(local_number_unreached);

        return { global_sum_distances, global_max_distance, global_sum_efficiency, global_number_unreached };
    }

private:
    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Always pushes a new distance
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs_push_brute_force(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
                                                mpiPP::RMAWindow<distance_type>& distances) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto buckets = Buckets(number_local_nodes);
        if (my_rank_cast == root_node.owning_mpi_rank) {
            buckets.move_node(root_node.node_id, 0);
            buckets.mark_as_active(root_node.node_id);
        }

        const auto process_bucket = [&graph, &distances, &buckets, delta, my_rank, my_rank_cast](const std::size_t /* current_epoch */) {
            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                using question_data_type = distance_type;

                auto question_function = [&active_nodes, &distances, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                    using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                    if (!active_nodes.contains(node_id)) {
                        return ResultType{};
                    }

                    auto return_value = ResultType{};
                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);
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

                mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function, answer_function);
            }
        };

        for (auto current_epoch = std::size_t{ 0 }; check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }

    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Checks via RMA if a newly calculated distance is lower than the current one; only pushes those values
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs_push_previous_check(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
                                                   mpiPP::RMAWindow<distance_type>& distances) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto buckets = Buckets(number_local_nodes);
        if (my_rank_cast == root_node.owning_mpi_rank) {
            buckets.move_node(root_node.node_id, 0);
            buckets.mark_as_active(root_node.node_id);
        }

        const auto process_bucket = [&graph, &distances, &buckets, delta, my_rank, my_rank_cast](const std::size_t /* current_epoch */) {
            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                using question_data_type = distance_type;

                auto question_function = [&active_nodes, &distances, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                    using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;

                    if (!active_nodes.contains(node_id)) {
                        return ResultType{};
                    }

                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = ResultType{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);

                        auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));
                        if (new_distance < old_distance) {
                            return_value.emplace_back(target_rank, target_node_id, new_distance);
                        }
                    }

                    return return_value;
                };

                auto answer_function = [&distances, &buckets, my_rank, my_rank_cast, delta](const node_id_type node_id, const question_data_type new_distance) {
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

                mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function, answer_function);
            }
        };

        for (auto current_epoch = std::size_t{ 0 }; check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }

    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Postpones updates of the distance to nodes that would fall outside of the current bucket
     *			based on long arcs to the last phase.
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs_push_short_long(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
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

            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                auto question_function = [&active_nodes, &distances, &long_arcs, delta, current_epoch, my_rank, my_rank_cast,
                                          &graph](const node_id_type node_id) {
                    using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                    if (!active_nodes.contains(node_id)) {
                        return ResultType{};
                    }

                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = ResultType{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);

                        auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));
                        if (old_distance <= new_distance) {
                            continue;
                        }

                        const auto target_bucket = compute_bucket_index(delta, old_distance);
                        if (target_bucket > current_epoch && utility::save_cast<distance_type>(weight) >= delta) {
                            // This is a long arc
                            long_arcs[node_id].emplace_back(target_rank, target_node_id, weight);
                            continue;
                        }

                        if (new_distance < old_distance) {
                            return_value.emplace_back(target_rank, target_node_id, new_distance);
                        }
                    }

                    return return_value;
                };

                mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function, answer_function);
            }

            auto question_function_long_arcs = [&distances, &long_arcs, my_rank](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                const auto long_arcs_it = long_arcs.find(node_id);
                if (long_arcs_it == long_arcs.end()) {
                    return ResultType{};
                }

                const auto& out_arcs = long_arcs_it->second;

                auto return_value = ResultType{};
                return_value.reserve(out_arcs.size());

                for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                    const auto my_distance = distances.get(node_id, my_rank);

                    const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);

                    auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));

                    if (new_distance < old_distance) {
                        return_value.emplace_back(target_rank, target_node_id, new_distance);
                    }
                }

                return return_value;
            };

            mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function_long_arcs, answer_function);
        };

        for (auto current_epoch = std::size_t{ 0 }; check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }

    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Postpones updates of the distance to nodes that would fall outside of the current bucket
     *			based on long or outer-short arcs to the last phase.
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs_push_inner_short(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
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

            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                auto question_function = [&active_nodes, &distances, &long_arcs, delta, current_epoch, my_rank, my_rank_cast,
                                          &graph](const node_id_type node_id) {
                    using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                    if (!active_nodes.contains(node_id)) {
                        return ResultType{};
                    }

                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = ResultType{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);

                        auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));
                        if (old_distance <= new_distance) {
                            continue;
                        }

                        const auto new_bucket = compute_bucket_index(delta, new_distance);
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

                mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function, answer_function);
            }

            auto question_function_long_arcs = [&distances, &long_arcs, my_rank](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                const auto long_arcs_it = long_arcs.find(node_id);
                if (long_arcs_it == long_arcs.end()) {
                    return ResultType{};
                }

                const auto& out_arcs = long_arcs_it->second;

                auto return_value = ResultType{};
                return_value.reserve(out_arcs.size());

                for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                    const auto my_distance = distances.get(node_id, my_rank);

                    const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);

                    auto old_distance = distances.get(target_node_id, mpiPP::MPIRank(target_rank));

                    if (new_distance < old_distance) {
                        return_value.emplace_back(target_rank, target_node_id, new_distance);
                    }
                }

                return return_value;
            };

            mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), question_function_long_arcs, answer_function);
        };

        for (auto current_epoch = std::size_t{ 0 }; check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }

    /**
     * @brief Compute the epochs and the phases of the delta-stepping algorithm. It has the following features:
     *		- Postpones updates of the distance to nodes that would fall outside of the current bucket
     *			based on long or outer-short arcs to the last phase.
     *		- Pulls in the last phase rather than pushes.
     * @param graph The distributed graph
     * @param root_node The root node of the SSSP
     * @param delta The delta (i.e., the width of the buckets)
     * @param distances The window that allows for accessing the distributedly-saved distances
     */
    static void compute_epochs_pull_model(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
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

            while (true) {
                auto active_nodes = buckets.get_active_nodes();

                const auto local_finished = active_nodes.empty();
                const auto all_finished = mpiPP::MPIReductions::all_reduce_and(local_finished);

                if (all_finished) {
                    break;
                }

                auto question_function = [&distances, delta, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                    using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<question_data_type, node_id_type>::result_type;
                    const auto my_distance = distances.get(node_id, my_rank);
                    const auto& out_arcs = graph.get_out_arcs(my_rank_cast, node_id);

                    auto return_value = ResultType{};
                    return_value.reserve(out_arcs.size());

                    for (const auto& [target_rank, target_node_id, weight] : out_arcs) {
                        if (utility::save_cast<distance_type>(weight) >= delta) {
                            // This is a long arc and will be processed by pulling
                            continue;
                        }

                        const auto new_distance = my_distance + utility::save_cast<distance_type>(weight);
                        return_value.emplace_back(target_rank, target_node_id, new_distance);
                    }

                    return return_value;
                };

                auto answer_function = [&distances, &buckets, my_rank, my_rank_cast, delta](const node_id_type node_id, const question_data_type new_distance) {
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

                mpiPP::comm_patterns::node_to_node_question<question_data_type>(graph.get_number_local_nodes(), active_nodes, question_function,
                                                                                answer_function);
            }

            auto question_function_pull = [&distances, &buckets, delta, current_epoch, my_rank, my_rank_cast, &graph](const node_id_type node_id) {
                using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<pull_question_data_type, node_id_type>::result_type;

                const auto bucket_index = buckets.get_current_bucket(node_id);
                if (bucket_index <= current_epoch) {
                    return ResultType{};
                }

                const auto my_distance = distances.get(node_id, my_rank);
                const auto& in_arcs = graph.get_in_arcs(my_rank_cast, node_id);

                auto return_value = ResultType{};
                return_value.reserve(in_arcs.size());

                for (const auto& [source_rank, source_node_id, weight] : in_arcs) {
                    if (utility::save_cast<distance_type>(weight) >= my_distance - current_epoch * delta) {
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
                    if (my_distance + utility::save_cast<distance_type>(weight) < question_distance) {
                        return my_distance + utility::save_cast<distance_type>(weight);
                    }

                    break;
                }

                return max_distance;
            };

            const auto& pull_answers = mpiPP::comm_patterns::node_to_node_question<pull_question_data_type, pull_answer_type>(
                graph.get_number_local_nodes(), question_function_pull, answer_function_pull);

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

        for (auto current_epoch = std::size_t{ 0 }; check_other_epoch_is_necessary(buckets, current_epoch); current_epoch++) {
            process_bucket(current_epoch);
        }
    }

    /**
     * @brief Checks if the ranks need another epoch
     * @param buckets The currently used buckets
     * @param planned_epoch The index of the next epoch
     * @return True iff the algorithm needs another epoch
     */
    [[nodiscard]] static bool check_other_epoch_is_necessary(const Buckets& buckets, const std::size_t planned_epoch) {
        const auto locally_need_another_epoch = buckets.has_nodes_left(planned_epoch);
        const auto globally_need_another_epoch = mpiPP::MPIReductions::all_reduce_or(locally_need_another_epoch);

        return globally_need_another_epoch;
    }

    /**
     * @brief Calculates the index of the bucket for a nodes with the given distance
     * @param delta The current delta
     * @param current_distance The given distance of a node
     * @return The index of the bucket
     */
    [[nodiscard]] static std::size_t compute_bucket_index(const distance_type delta, const distance_type current_distance) {
        const auto index = current_distance / delta;
        return index;
    }
};
