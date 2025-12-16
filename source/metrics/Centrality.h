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
#include "metrics/Distance.h"
#include "metrics/NodeCounter.h"
#include "utility/PoolAllocator.h"
#include "utility/Status.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/data/prefix_sum.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"
#include "mpi-wrapper/MPISynchronization.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <utility>
#include <vector>

class BetweennessCentrality {
public:
    /**
     * @brief Calculates the average betweenness centrality in the graph, i.e., for all nodes i and j,
     *		looks at all nodes on the paths p1, ..., pk from i to j with minimal length.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V * (A + V log V)), neglecting MPI collectives.
     * @param graph The graph
     * @return The betweenness centrality averaged across all nodes in the network
     */
    [[nodiscard]] static double compute_average_betweenness_centrality(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        auto status = Status{ number_local_nodes, "Betweenness Centrality" };

        auto local_betweenness = std::vector<double>(total_number_nodes, 0.0);
        auto distances = std::vector<distance_type>(total_number_nodes);

        auto shortest_paths = std::vector<std::vector<NodePath, PoolAllocator<NodePath>>>(total_number_nodes);
        for (auto& shorted_path : shortest_paths) {
            shorted_path.reserve(16);
        }

        for (auto node_id = 0U; node_id < number_local_nodes; ++node_id) {
            compute_betweenness_centrality(graph, node_id, prefix_distribution, local_betweenness, distances, shortest_paths);
            status.report(node_id);
        }

        mpiPP::MPISynchronization::barrier();

        status.finish();

        const auto result = mpiPP::MPIReductions::reduce_componentwise_sum(local_betweenness);

        const auto betweenness_centrality = std::reduce(result.begin(), result.end(), 0.0, std::plus<double>{});
        const auto average_betweenness_centrality = betweenness_centrality / static_cast<double>(total_number_nodes);

        return average_betweenness_centrality;
    }

private:
    static void compute_betweenness_centrality(const DistributedGraph& graph, unsigned int node_id, const std::vector<std::uint32_t>& prefix_distribution,
                                               std::vector<double>& local_betweenness, std::vector<distance_type>& distances,
                                               std::vector<std::vector<NodePath, PoolAllocator<NodePath>>>& shortest_paths) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        for (auto& shortest_path : shortest_paths) {
            shortest_path.clear();
        }

        std::ranges::fill(distances, std::numeric_limits<distance_type>::max());

        auto queue_space = std::vector<VertexDistancePath>{};
        queue_space.reserve(1000);
        auto shortest_paths_queue = std::priority_queue<VertexDistancePath, std::vector<VertexDistancePath>, GreaterDistance>{ GreaterDistance{}, std::move(queue_space) };

        const auto root_id = prefix_distribution[utility::save_cast<std::size_t>(my_rank)] + node_id;

        distances[root_id] = 0;

        auto start = NodePath(10);
        start.append_node(node_id);

        shortest_paths[root_id] = { start };
        shortest_paths_queue.emplace(my_rank, node_id, 0, std::move(start));

        while (!shortest_paths_queue.empty()) {
            const auto vdp = std::move(const_cast<VertexDistancePath&>(shortest_paths_queue.top()));
            shortest_paths_queue.pop();

            const auto& [current_rank, current_id, current_distance, current_path] = vdp;

            const auto out_arcs = graph.get_out_arcs(current_rank, current_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto new_distance = current_distance + utility::save_cast<distance_type>(std::abs(weight));

                const auto other_node_id = prefix_distribution[utility::save_cast<std::size_t>(target_rank)] + target_id;
                if (distances[other_node_id] < new_distance) {
                    continue;
                }

                if (distances[other_node_id] > new_distance) {
                    distances[other_node_id] = new_distance;
                    shortest_paths[other_node_id].clear();
                }

                auto new_path = current_path;
                new_path.append_node(other_node_id);

                shortest_paths_queue.emplace(target_rank, target_id, new_distance, new_path);
                shortest_paths[other_node_id].emplace_back(std::move(new_path));
            }
        }

        for (const auto& shortest_paths_to_node : shortest_paths) {
            const auto number_shortest_paths = shortest_paths_to_node.size();
            if (number_shortest_paths == 0) {
                continue;
            }

            const auto portion = 1.0 / static_cast<double>(number_shortest_paths);

            for (const auto& shortest_path_to_node : shortest_paths_to_node) {
                const auto nodes_on_path = shortest_path_to_node.get_nodes();

                for (const auto id : std::span{ nodes_on_path }.subspan(1, nodes_on_path.size() - 2)) {
                    local_betweenness[id] += portion;
                }
            }
        }
    }
};
