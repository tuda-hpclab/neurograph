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
#include "metrics/paths/DijkstraVariants/DijkstraCommon.h"
#include "metrics/paths/types/VertexDistance.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>

#include <cstddef>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief Betweenness centrality variant after Brandes (2001): per source, a Dijkstra pass counts
 *		the number of shortest paths to every node and records the predecessor arcs of the
 *		shortest-path DAG; a backward sweep over the settled nodes in order of decreasing distance
 *		then accumulates every node's pair dependencies.
 *		No path is ever materialized, so one source costs O(A + V log V) time and O(V + A) memory
 *		even when the number of shortest paths grows exponentially with the number of nodes.
 *		Computes exactly the same values as PathEnumerationVariant and shares its assumption of
 *		non-zero arc weights: a zero-length arc would break the settling order the sweep relies on.
 */
class BrandesVariant {
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
        // The number of shortest paths can overflow any integer type, so it is counted in doubles:
        // exact up to 2^53 paths, and losing only precision (not correctness) beyond that.
        auto number_shortest_paths = std::vector<double>(total_number_nodes, 0.0);
        auto dependencies = std::vector<double>(total_number_nodes, 0.0);
        auto predecessors = std::vector<std::vector<global_node_id_type>>(total_number_nodes);

        auto settled_nodes = std::vector<global_node_id_type>{};
        settled_nodes.reserve(total_number_nodes);

        auto queue_space = std::vector<VertexDistance<distance_type>>{};
        queue_space.reserve(1000);
        // Every search drains the queue completely, so one queue serves all sources and keeps its storage.
        auto shortest_paths_queue = ShortestPathsQueue{ GreaterVertexDistance<distance_type>{}, std::move(queue_space) };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            accumulate_from_source(graph, node_id, prefix_distribution, local_betweenness, distances, number_shortest_paths, dependencies, predecessors, settled_nodes, shortest_paths_queue);
            status.report(node_id);
        }
    }

private:
    using ShortestPathsQueue = std::priority_queue<VertexDistance<distance_type>, std::vector<VertexDistance<distance_type>>, GreaterVertexDistance<distance_type>>;

    static void accumulate_from_source(const DistributedGraph& graph, const node_id_type node_id, const std::span<const global_node_id_type> prefix_distribution,
                                       std::vector<double>& local_betweenness, std::vector<distance_type>& distances, std::vector<double>& number_shortest_paths,
                                       std::vector<double>& dependencies, std::vector<std::vector<global_node_id_type>>& predecessors,
                                       std::vector<global_node_id_type>& settled_nodes, ShortestPathsQueue& shortest_paths_queue) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        // The previous source only dirtied the entries of the nodes it reached, and every reached node
        // was settled: the queue always drains completely, so every relaxed node is popped and settled
        // eventually, and the sweep writes dependencies only for settled nodes and their (settled)
        // predecessors. Resetting exactly those entries restores the clean state (infinite distance,
        // zero paths and dependencies, no predecessors) everywhere without an O(V) sweep per source.
        for (const auto settled_node : settled_nodes) {
            distances[settled_node] = std::numeric_limits<distance_type>::max();
            number_shortest_paths[settled_node] = 0.0;
            dependencies[settled_node] = 0.0;
            predecessors[settled_node].clear();
        }
        settled_nodes.clear();

        const auto root_id = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)] + node_id;

        distances[root_id] = 0;
        number_shortest_paths[root_id] = 1.0;
        shortest_paths_queue.emplace(my_rank, node_id, 0);

        while (!shortest_paths_queue.empty()) {
            const auto [current_rank, current_id, current_distance] = shortest_paths_queue.top();
            shortest_paths_queue.pop();

            const auto current_node_id = prefix_distribution[utility::safe_cast<std::size_t>(current_rank)] + current_id;
            if (distances[current_node_id] < current_distance) {
                // Stale queue entry: a shorter distance to this node was found after it had been enqueued.
                continue;
            }

            // The node is settled now, exactly once per search: re-inserts require a strict distance
            // improvement, so only the entry carrying the final distance survives the check above.
            // Its number of shortest paths is final as well, because every predecessor on a shortest
            // path has a strictly smaller distance (non-zero weights) and was settled and expanded before.
            settled_nodes.push_back(current_node_id);

            const auto out_arcs = graph.get_out_arcs(current_rank, current_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto new_distance = current_distance + DijkstraCommon::transform_weight<distance_type>(weight);

                const auto other_node_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;

                if (new_distance < distances[other_node_id]) {
                    distances[other_node_id] = new_distance;
                    number_shortest_paths[other_node_id] = number_shortest_paths[current_node_id];
                    predecessors[other_node_id].clear();
                    predecessors[other_node_id].emplace_back(current_node_id);
                    shortest_paths_queue.emplace(target_rank, target_id, new_distance);
                } else if (new_distance == distances[other_node_id]) {
                    // Another bundle of shortest paths to the target, all running through the current
                    // node. Parallel arcs add one predecessor entry each, matching the paths that
                    // PathEnumerationVariant would enumerate.
                    number_shortest_paths[other_node_id] += number_shortest_paths[current_node_id];
                    predecessors[other_node_id].emplace_back(current_node_id);
                }
            }
        }

        // Accumulate the dependencies backwards through the shortest-path DAG, i.e., in order of
        // decreasing distance: delta(v) = sum over successors w of sigma(v) / sigma(w) * (1 + delta(w)).
        // The root is always settled first, so stopping at index 1 both skips it as a crediting node
        // (a source gains no betweenness from its own paths) and keeps sigma > 0 in the division.
        for (auto index = settled_nodes.size(); index > 1;) {
            --index;
            const auto settled_node = settled_nodes[index];

            const auto dependency_portion = (1.0 + dependencies[settled_node]) / number_shortest_paths[settled_node];

            for (const auto predecessor : predecessors[settled_node]) {
                dependencies[predecessor] += number_shortest_paths[predecessor] * dependency_portion;
            }

            local_betweenness[settled_node] += dependencies[settled_node];
        }
    }
};
