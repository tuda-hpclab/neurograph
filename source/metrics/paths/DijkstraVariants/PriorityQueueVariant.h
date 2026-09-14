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
#include "metrics/paths/DijkstraVariants/DijkstraCommon.h"
#include "metrics/paths/types/VertexDistance.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief Dijkstra vertex queue variant that uses a binary heap (std::priority_queue) with lazy deletion:
 *		instead of decreasing keys in place, improved nodes are re-inserted and stale entries are
 *		skipped when popped. Complexity is O((V + A) log V), neglecting MPI communication.
 */
class PriorityQueueVariant {
public:
    /**
     * @brief Runs the Dijkstra main loop starting at the calling rank's root_node_id.
     * @param graph The graph
     * @param root_node_id The starting vertex, owned by the calling MPI rank
     * @param prefix_distribution The prefix sum of the number of nodes on each MPI rank
     * @param distances One entry per global node; pre-filled with the maximum value and 0 at the root
     * @tparam d_type The type for the distances, either distance_type or inverse_distance_type
     */
    template <typename d_type>
    static void compute_distances(const DistributedGraph& graph, const node_id_type root_node_id,
                                  const std::span<const global_node_id_type> prefix_distribution, std::vector<d_type>& distances) {
        constexpr auto max_distance = std::numeric_limits<d_type>::max();

        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto queue_space = std::vector<VertexDistance<d_type>>{};
        queue_space.reserve(200);

        auto shortest_paths_queue =
            std::priority_queue<VertexDistance<d_type>, std::vector<VertexDistance<d_type>>, GreaterVertexDistance<d_type>>{ GreaterVertexDistance<d_type>{}, std::move(queue_space) };

        shortest_paths_queue.emplace(my_rank, root_node_id, 0);

        while (!shortest_paths_queue.empty()) {
            const auto [current_rank, current_id, current_distance] = shortest_paths_queue.top();
            shortest_paths_queue.pop();

            const auto current_node_id = prefix_distribution[utility::safe_cast<std::size_t>(current_rank)] + current_id;
            if (distances[current_node_id] < current_distance) {
                // Stale queue entry: a shorter distance to this node was found after it had been enqueued.
                continue;
            }

            const auto out_arcs = graph.get_out_arcs(current_rank, current_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto arc_distance = DijkstraCommon::transform_weight<d_type>(weight);
                if (arc_distance == max_distance) {
                    // An infinitely long arc (zero weight in the inverse metric) can never shorten a path;
                    // adding it to current_distance would collide with the unreachable sentinel.
                    continue;
                }

                const auto new_distance = current_distance + arc_distance;

                const auto other_node_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;

                if (distances[other_node_id] <= new_distance) {
                    continue;
                }

                distances[other_node_id] = new_distance;
                shortest_paths_queue.emplace(target_rank, target_id, new_distance);
            }
        }
    }
};
