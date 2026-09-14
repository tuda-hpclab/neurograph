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
#include "metrics/paths/types/NodePath.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/memory/PoolAllocator.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief Betweenness centrality variant that materializes every shortest path explicitly:
 *		a Dijkstra-like search whose queue entries each carry the full path from the root, and
 *		every shortest path to every node is stored as the list of its nodes.
 *		Since the number of shortest paths between two nodes can grow exponentially with the
 *		number of nodes, time and memory are exponential in the worst case; only when the
 *		shortest paths are (nearly) unique does one source cost O(A + V log V).
 *		Assumes non-zero arc weights: a zero-weight cycle makes the enumeration diverge.
 */
class PathEnumerationVariant {
public:
    /**
     * @brief Accumulates the betweenness contributions of all sources owned by the calling MPI rank.
     * @param graph The graph
     * @param prefix_distribution The prefix sum of the number of nodes on each MPI rank
     * @param local_betweenness One entry per global node; the contributions are added onto it
     * @param status The progress reporter; reported once per local source
     */
    static void accumulate_betweenness(const DistributedGraph& graph, const std::span<const global_node_id_type> prefix_distribution,
                                       std::vector<double>& local_betweenness, mpiPP::MPIProgress& status) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = local_betweenness.size();

        auto distances = std::vector<distance_type>(total_number_nodes, std::numeric_limits<distance_type>::max());

        auto shortest_paths = std::vector<std::vector<NodePath, utility::PoolAllocator<NodePath>>>(total_number_nodes);
        for (auto& shortest_path : shortest_paths) {
            shortest_path.reserve(16);
        }

        auto touched_nodes = std::vector<global_node_id_type>{};
        touched_nodes.reserve(total_number_nodes);

        auto queue_space = std::vector<VertexDistancePath>{};
        queue_space.reserve(1000);
        // Every search drains the queue completely, so one queue serves all sources and keeps its storage.
        auto shortest_paths_queue = ShortestPathsQueue{ GreaterDistance{}, std::move(queue_space) };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            accumulate_from_source(graph, node_id, prefix_distribution, local_betweenness, distances, shortest_paths, touched_nodes, shortest_paths_queue);
            status.report(node_id);
        }
    }

private:
    using ShortestPathsQueue = std::priority_queue<VertexDistancePath, std::vector<VertexDistancePath>, GreaterDistance>;

    static void accumulate_from_source(const DistributedGraph& graph, const node_id_type node_id, const std::span<const global_node_id_type> prefix_distribution,
                                       std::vector<double>& local_betweenness, std::vector<distance_type>& distances,
                                       std::vector<std::vector<NodePath, utility::PoolAllocator<NodePath>>>& shortest_paths, std::vector<global_node_id_type>& touched_nodes,
                                       ShortestPathsQueue& shortest_paths_queue) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        // The previous source only dirtied the entries of the nodes it reached, recorded in
        // touched_nodes; resetting exactly those restores the clean state (infinite distance,
        // no stored paths) everywhere without an O(V) sweep per source.
        for (const auto touched_node : touched_nodes) {
            distances[touched_node] = std::numeric_limits<distance_type>::max();
            shortest_paths[touched_node].clear();
        }
        touched_nodes.clear();

        const auto root_id = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)] + node_id;

        distances[root_id] = 0;
        touched_nodes.push_back(root_id);

        auto start = NodePath(10);
        start.append_node(root_id);

        shortest_paths[root_id] = { start };
        shortest_paths_queue.emplace(my_rank, node_id, 0, std::move(start));

        while (!shortest_paths_queue.empty()) {
            const auto vdp = std::move(const_cast<VertexDistancePath&>(shortest_paths_queue.top()));
            shortest_paths_queue.pop();

            const auto& [current_rank, current_id, current_distance, current_path] = vdp;

            const auto current_node_id = prefix_distribution[utility::safe_cast<std::size_t>(current_rank)] + current_id;
            if (distances[current_node_id] < current_distance) {
                // Stale queue entry: a strictly shorter distance to this node was found after this path
                // had been enqueued. All shorter paths were popped and expanded before this one, so every
                // extension would fail the distance check below anyway; skip the arc scan altogether.
                // Alternative paths of exactly the shortest distance still pass and get extended.
                continue;
            }

            const auto out_arcs = graph.get_out_arcs(current_rank, current_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                // std::abs(weight) is undefined behavior for the most negative weight_type value,
                // so widen to a larger signed type before taking the absolute value.
                const auto new_distance = current_distance + utility::safe_cast<distance_type>(std::abs(static_cast<std::int64_t>(weight)));

                const auto other_node_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;
                if (distances[other_node_id] < new_distance) {
                    continue;
                }

                if (distances[other_node_id] > new_distance) {
                    if (distances[other_node_id] == std::numeric_limits<distance_type>::max()) {
                        // The node is reached for the first time from this source.
                        touched_nodes.push_back(other_node_id);
                    }

                    distances[other_node_id] = new_distance;
                    shortest_paths[other_node_id].clear();
                }

                auto new_path = current_path;
                new_path.append_node(other_node_id);

                shortest_paths_queue.emplace(target_rank, target_id, new_distance, new_path);
                shortest_paths[other_node_id].emplace_back(std::move(new_path));
            }
        }

        for (const auto touched_node : touched_nodes) {
            const auto& shortest_paths_to_node = shortest_paths[touched_node];

            // Every touched node stores at least one path: a path is appended right after the touch,
            // and later improvements replace the stored paths instead of leaving the list empty.
            const auto portion = 1.0 / static_cast<double>(shortest_paths_to_node.size());

            for (const auto& shortest_path_to_node : shortest_paths_to_node) {
                const auto nodes_on_path = shortest_path_to_node.get_nodes();
                if (nodes_on_path.size() <= 2) {
                    // Only the source and the target lie on the path; there are no interior nodes to credit.
                    continue;
                }

                for (const auto id : std::span{ nodes_on_path }.subspan(1, nodes_on_path.size() - 2)) {
                    local_betweenness[id] += portion;
                }
            }
        }
    }
};
