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
#include "metrics/Dijkstra.h"
#include "metrics/NodeCounter.h"
#include "utility/Status.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/data/prefix_sum.hpp"

#include "mpi-wrapper/MPIAdvancedReductions.h"
#include "mpi-wrapper/MPIReductions.h"
#include "mpi-wrapper/MPISynchronization.h"

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
     * @return A quintuple of
     *		(1) the average shortest path length in the whole graph
     *		(2) the average efficiency of those paths
     *		(3) the diameter of the graph
     *		(4) the number of disconnected pairs (i.e., how many pairs (v, w) exists such that there is no path from v to w)
     *		(5) the sizes of the components, i.e., (5)[i] == j indicates that there are j nodes in components of size i
     */
    [[nodiscard]] static apsp_global_result<distance_type> compute_apsp(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        auto status = Status{ number_local_nodes, "APSP" };

        auto sum_shortest_path_locally = distance_type{ 0 };
        auto sum_efficiency_locally = 0.0;
        auto maximum_distance = distance_type{ 0 };
        auto number_unreachables_locally = global_node_id_type{ 0 };

        auto cluster_sizes = std::unordered_map<global_node_id_type, global_node_id_type>{};
        cluster_sizes.reserve(200);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            auto distances = std::vector<distance_type>(total_number_nodes);
            const auto [sum_shortest_path_from_node, max_dist, sum_efficiency_from_node, number_unreachables_from_node] = Dijkstra::compute_sssp<distance_type>(graph, node_id, prefix_distribution, distances);

            {
                sum_shortest_path_locally += sum_shortest_path_from_node;
                sum_efficiency_locally += sum_efficiency_from_node;
                number_unreachables_locally += number_unreachables_from_node;
                maximum_distance = std::max(max_dist, maximum_distance);

                const auto number_reachable_nodes = total_number_nodes - number_unreachables_from_node;
                ++cluster_sizes[number_reachable_nodes];

                status.report(node_id);
            }
        }

        mpiPP::MPISynchronization::barrier();

        status.finish();

        const auto sum_shortest_paths = mpiPP::MPIReductions::reduce_sum(sum_shortest_path_locally);
        const auto sum_efficiency = mpiPP::MPIReductions::reduce_sum(sum_efficiency_locally);
        const auto diameter = mpiPP::MPIReductions::reduce_max(maximum_distance);
        const auto sum_unreachables = mpiPP::MPIReductions::reduce_sum(number_unreachables_locally);

        const auto number_pairs_without_self = (total_number_nodes * total_number_nodes) - total_number_nodes;
        const auto number_reached_pairs = number_pairs_without_self - number_unreachables_locally;

        const auto average_shortest_path_length = static_cast<double>(sum_shortest_paths) / static_cast<double>(number_reached_pairs);
        const auto average_efficiency = sum_efficiency / static_cast<double>(number_reached_pairs);

        const auto global_cluster_sizes = mpiPP::MPIAdvancedReductions::reduce_map(cluster_sizes);

        return { average_shortest_path_length, average_efficiency, diameter, sum_unreachables, global_cluster_sizes };
    }

    /**
     * @brief Performs a repeated Dijkstra's algorithm to determine the average shortest path length,
     *		as well as the average efficiency of those paths, the number of disconnected pairs,
     *		and the size distribution of those disconnected clusters.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V * (A + V log V)), neglecting MPI collectives.
     *      Uses the inverse of the weights of the arcs
     * @param graph The graph
     * @return A quintuple of
     *		(1) the average shortest path length in the whole graph
     *		(2) the average efficiency of those paths
     *		(3) the diameter of the graph
     *		(4) the number of disconnected pairs (i.e., how many pairs (v, w) exists such that there is no path from v to w)
     *		(5) the sizes of the components, i.e., (5)[i] == j indicates that there are j nodes in components of size i
     */
    [[nodiscard]] static apsp_global_result<inverse_distance_type> compute_apsp_inverse(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        auto status = Status{ number_local_nodes, "APSP" };

        auto sum_shortest_path_locally = inverse_distance_type{ 0 };
        auto sum_efficiency_locally = 0.0;
        auto maximum_distance = inverse_distance_type{ 0 };
        auto number_unreachables_locally = global_node_id_type{ 0 };

        auto distances = std::vector<inverse_distance_type>(total_number_nodes);
        auto cluster_sizes = std::unordered_map<global_node_id_type, global_node_id_type>{};
        cluster_sizes.reserve(200);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto [sum_shortest_path_from_node, max_dist, sum_efficiency_from_node, number_unreachables_from_node] = Dijkstra::compute_sssp<inverse_distance_type>(graph, node_id, prefix_distribution, distances);

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
        const auto number_reached_pairs = number_pairs_without_self - number_unreachables_locally;

        const auto average_shortest_path_length = static_cast<double>(sum_shortest_paths) / static_cast<double>(number_reached_pairs);
        const auto average_efficiency = sum_efficiency / static_cast<double>(number_reached_pairs);

        const auto global_cluster_sizes = mpiPP::MPIAdvancedReductions::reduce_map(cluster_sizes);

        return { average_shortest_path_length, average_efficiency, diameter, sum_unreachables, global_cluster_sizes };
    }
};
