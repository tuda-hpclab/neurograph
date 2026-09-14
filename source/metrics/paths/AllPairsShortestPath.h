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
#include "metrics/counting/NodeCounter.h"
#include "metrics/counting/NodeDistributionCounter.h"
#include "metrics/paths/DeltaStepping.h"
#include "metrics/paths/Dijkstra.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/data/prefix_sum.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/reductions/MPIAdvancedReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

class AllPairsShortestPath {
public:
    /**
     * @brief Performs a repeated Dijkstra's algorithm to determine the average shortest path length,
     *		as well as the average efficiency of those paths, the number of disconnected pairs,
     *		and the size distribution of those disconnected clusters.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V * (A + V log V)), neglecting MPI collectives.
     * @param graph The graph
     * @param queue_type Which vertex queue implementation Dijkstra uses (see DijkstraQueueType)
     * @return A quintuple of
     *		(1) the average shortest path length in the whole graph
     *		(2) the average efficiency of those paths
     *		(3) the diameter of the graph
     *		(4) the number of disconnected pairs (i.e., how many pairs (v, w) exists such that there is no path from v to w)
     *		(5) the sizes of the components, i.e., (5)[i] == j indicates that there are j nodes in components of size i
     */
    [[nodiscard]] static ApspGlobalResult<distance_type> compute_apsp(const DistributedGraph& graph,
                                                                       const DijkstraQueueType queue_type = DijkstraQueueType::PriorityQueue) {
        return compute_apsp_impl<distance_type>(graph, queue_type);
    }

    /**
     * @brief Performs a repeated Dijkstra's algorithm to determine the average shortest path length,
     *		as well as the average efficiency of those paths, the number of disconnected pairs,
     *		and the size distribution of those disconnected clusters.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V * (A + V log V)), neglecting MPI collectives.
     *      Uses the inverse of the weights of the arcs
     * @param graph The graph
     * @param queue_type Which vertex queue implementation Dijkstra uses; BucketQueue is not valid here
     *		(the inverse distances are not integers) and makes the computation throw
     * @return A quintuple of
     *		(1) the average shortest path length in the whole graph
     *		(2) the average efficiency of those paths
     *		(3) the diameter of the graph
     *		(4) the number of disconnected pairs (i.e., how many pairs (v, w) exists such that there is no path from v to w)
     *		(5) the sizes of the components, i.e., (5)[i] == j indicates that there are j nodes in components of size i
     */
    [[nodiscard]] static ApspGlobalResult<inverse_distance_type> compute_apsp_inverse(const DistributedGraph& graph,
                                                                                       const DijkstraQueueType queue_type = DijkstraQueueType::PriorityQueue) {
        return compute_apsp_impl<inverse_distance_type>(graph, queue_type);
    }

    /**
     * @brief Performs repeated delta-stepping SSSP computations to determine the average shortest path length,
     *		as well as the average efficiency of those paths, the number of disconnected pairs,
     *		and the size distribution of those disconnected clusters.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Unlike compute_apsp, every rank participates in every single-source computation
     *		(delta-stepping is itself a collective algorithm across all ranks; see compute_apsp_impl_delta_stepping),
     *		so there is only a distance_type overload: delta-stepping's bucket index (distance / delta) assumes
     *		an integer distance and is not meaningful for inverse_distance_type.
     * @param graph The graph
     * @param delta The delta (i.e., the width of the buckets) used by the delta-stepping computation
     * @param epoch_type Which delta-stepping epoch strategy to use
     * @return See compute_apsp
     */
    [[nodiscard]] static ApspGlobalResult<distance_type> compute_apsp_delta_stepping(const DistributedGraph& graph, const distance_type delta,
                                                                                      const DeltaSteppingEpochType epoch_type) {
        return compute_apsp_impl_delta_stepping(graph, delta, epoch_type);
    }

private:
    /**
     * @brief Shared implementation of the regular and the inverse all-pairs shortest path computation.
     *		The distance type selects the metric: distance_type uses the arc weights, inverse_distance_type
     *		uses their inverse (see Dijkstra::compute_sssp).
     * @tparam d_type The type for the distances, either distance_type or inverse_distance_type
     * @param graph The graph
     * @param queue_type Which vertex queue implementation Dijkstra uses (see DijkstraQueueType)
     * @return The aggregated all-pairs result (see compute_apsp)
     */
    template <typename d_type>
    [[nodiscard]] static ApspGlobalResult<d_type> compute_apsp_impl(const DistributedGraph& graph, const DijkstraQueueType queue_type) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        auto status = mpiPP::MPIProgress{ number_local_nodes, "APSP" };

        auto sum_shortest_path_locally = d_type{ 0 };
        auto sum_efficiency_locally = 0.0;
        auto maximum_distance = d_type{ 0 };
        auto number_unreachables_locally = global_node_id_type{ 0 };

        auto distances = std::vector<d_type>(total_number_nodes);
        auto cluster_sizes = std::unordered_map<global_node_id_type, global_node_id_type>{};
        cluster_sizes.reserve(200);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto [sum_shortest_path_from_node, max_dist, sum_efficiency_from_node, number_unreachables_from_node] =
                Dijkstra::compute_sssp<d_type>(graph, node_id, prefix_distribution, distances, queue_type);

            sum_shortest_path_locally += sum_shortest_path_from_node;
            sum_efficiency_locally += sum_efficiency_from_node;
            number_unreachables_locally += number_unreachables_from_node;
            maximum_distance = std::max(max_dist, maximum_distance);

            const auto number_reachable_nodes = total_number_nodes - number_unreachables_from_node;
            ++cluster_sizes[number_reachable_nodes];

            status.report(node_id);
        }

        mpiPP::MPISynchronization::barrier();

        status.finish();

        const auto sum_shortest_paths = mpiPP::MPIReductions::reduce_sum(sum_shortest_path_locally);
        const auto sum_efficiency = mpiPP::MPIReductions::reduce_sum(sum_efficiency_locally);
        const auto diameter = mpiPP::MPIReductions::reduce_max(maximum_distance);
        const auto sum_unreachables = mpiPP::MPIReductions::reduce_sum(number_unreachables_locally);

        const auto number_pairs_without_self = (total_number_nodes * total_number_nodes) - total_number_nodes;
        const auto number_reached_pairs = number_pairs_without_self - sum_unreachables;

        const auto average_shortest_path_length = static_cast<double>(sum_shortest_paths) / static_cast<double>(number_reached_pairs);
        const auto average_efficiency = sum_efficiency / static_cast<double>(number_reached_pairs);

        const auto global_cluster_sizes = mpiPP::MPIAdvancedReductions::reduce_map(cluster_sizes);

        return { average_shortest_path_length, average_efficiency, diameter, sum_unreachables, global_cluster_sizes };
    }

    /**
     * @brief Maps a global node id to the NodeIdentifier (owning rank + local id) of the node it refers to.
     *		Inverse of the "prefix_distribution[rank] + local_id" mapping used throughout this file.
     * @param prefix_distribution The exclusive prefix sum of the number of nodes on each MPI rank
     * @param global_id The global node id to resolve
     * @return The NodeIdentifier of the node with that global id
     */
    [[nodiscard]] static NodeIdentifier find_owning_node_identifier(const std::span<const global_node_id_type> prefix_distribution,
                                                                     const global_node_id_type global_id) {
        const auto rank_it = std::ranges::upper_bound(prefix_distribution, global_id);
        const auto rank_index = (rank_it - prefix_distribution.begin()) - 1;

        const auto rank = utility::safe_cast<mpi_rank_type>(rank_index);
        const auto local_id = utility::safe_cast<node_id_type>(global_id - prefix_distribution[utility::safe_cast<std::size_t>(rank_index)]);

        return { rank, local_id };
    }

    /**
     * @brief Delta-stepping-based implementation of the all-pairs shortest path computation.
     *		Unlike compute_apsp_impl, this cannot loop over only the locally-owned nodes of each rank:
     *		DeltaStepping::compute_shortest_distances is itself a collective operation that every rank
     *		must enter together for the same root node, and it already returns the fully cross-rank-reduced
     *		result for that one source. So every rank loops over all total_number_nodes global sources
     *		together, and the running sums are used directly afterwards without a further MPI reduction
     *		(reducing already-identical, already-global per-source results again would multiply them by
     *		the rank count).
     * @param graph The graph
     * @param delta The delta (i.e., the width of the buckets)
     * @param epoch_type Which delta-stepping epoch strategy to use
     * @return The aggregated all-pairs result (see compute_apsp)
     */
    [[nodiscard]] static ApspGlobalResult<distance_type> compute_apsp_impl_delta_stepping(const DistributedGraph& graph, const distance_type delta,
                                                                                           const DeltaSteppingEpochType epoch_type) {
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        // Every rank runs the same total_number_nodes iterations here (see the class-level note above), so
        // only rank 0's iteration count must flow into the status reporter's collective reduction, or the
        // displayed total would be inflated by the rank count.
        const auto local_iterations_for_status = mpiPP::MPIInfo::is_root_rank() ? total_number_nodes : global_node_id_type{ 0 };
        auto status = mpiPP::MPIProgress{ local_iterations_for_status, "APSP (Delta-Stepping)" };

        auto sum_shortest_paths = distance_type{ 0 };
        auto sum_efficiency = 0.0;
        auto diameter = distance_type{ 0 };
        auto number_disconnected_pairs = global_node_id_type{ 0 };

        auto cluster_sizes = std::unordered_map<global_node_id_type, global_node_id_type>{};
        cluster_sizes.reserve(200);

        for (auto global_source_id = global_node_id_type{ 0 }; global_source_id < total_number_nodes; ++global_source_id) {
            const auto root = find_owning_node_identifier(prefix_distribution, global_source_id);

            const auto [sum_from_node, max_distance_from_node, efficiency_from_node, number_unreachable_from_node] =
                DeltaStepping::compute_shortest_distances(graph, root, delta, epoch_type);

            sum_shortest_paths += sum_from_node;
            sum_efficiency += efficiency_from_node;
            diameter = std::max(diameter, max_distance_from_node);
            number_disconnected_pairs += number_unreachable_from_node;

            const auto number_reachable_nodes = total_number_nodes - number_unreachable_from_node;
            ++cluster_sizes[number_reachable_nodes];

            status.report(global_source_id);
        }

        mpiPP::MPISynchronization::barrier();

        status.finish();

        const auto number_pairs_without_self = (total_number_nodes * total_number_nodes) - total_number_nodes;
        const auto number_reached_pairs = number_pairs_without_self - number_disconnected_pairs;

        const auto average_shortest_path_length = static_cast<double>(sum_shortest_paths) / static_cast<double>(number_reached_pairs);
        const auto average_efficiency = sum_efficiency / static_cast<double>(number_reached_pairs);

        return { average_shortest_path_length, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes };
    }
};
