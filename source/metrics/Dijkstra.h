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

#include "mpi-wrapper/MPIInfo.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

class Dijkstra {
public:
    /**
     * @brief Performs Dijkstra's algorithm on graph starting at node_id.
     *		Complexity is O(A + V log V), neglecting MPI collectives.
     *      Changes the behavior depending on d_type:
     *          - distance_type: Uses the weight of the arcs as distances
     *          - inverse_distance_type: Uses 1/weight of the arcs as distances
     * @param graph The graph
     * @param node_id The starting vertex
     * @param prefix_distribution The prefix sum of the number of nodes on each MPI rank
     * @param distances A buffer that holds the distances; only allocated once and re-used
     * @tparam d_type The type for the distances, can be distance_type (default) or inverse_distance_type
     * @return A triple of
     *		(1) the sum of all found shortest-path distances,
     *		(2) the sum of all efficiencies of the shortest paths,
     *		(3) the number of unreachable nodes
     */
    template <typename d_type = distance_type>
    [[nodiscard]] static sssp_result<d_type> compute_sssp(const DistributedGraph& graph, const node_id_type node_id,
                                                          const std::span<const global_node_id_type> prefix_distribution, std::vector<d_type>& distances) {
        constexpr auto max_distance = std::numeric_limits<d_type>::max();

        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto queue_space = std::vector<VertexDistance<d_type>>{};
        queue_space.reserve(200);

        auto shortest_paths_queue = std::priority_queue<VertexDistance<d_type>, std::vector<VertexDistance<d_type>>, std::greater<>>{ std::greater{}, std::move(queue_space) };

        std::ranges::fill(distances, max_distance);

        const auto root_id = prefix_distribution[utility::save_cast<std::size_t>(my_rank)] + node_id;
        distances[root_id] = 0;

        shortest_paths_queue.emplace(my_rank, node_id, 0);

        const auto transform_weight = [](const weight_type weight) -> d_type {
            constexpr bool v1 = std::is_same_v<d_type, distance_type>;
            constexpr bool v2 = std::is_same_v<d_type, inverse_distance_type>;

            if constexpr (v1) {
                return utility::save_cast<d_type>(std::abs(weight));
            }

            if constexpr (v2) {
                return 1.0 / static_cast<inverse_distance_type>(weight);
            }

            if constexpr (!v1 && !v2) {
                // GCC 11 falsly triggers here
                // static_assert(false);
            }

            return d_type{ 0 };
        };

        while (!shortest_paths_queue.empty()) {
            const auto [current_rank, current_id, current_distance] = shortest_paths_queue.top();
            shortest_paths_queue.pop();

            const auto out_arcs = graph.get_out_arcs(current_rank, current_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto new_distance = current_distance + transform_weight(weight);

                const auto other_node_id = prefix_distribution[utility::save_cast<std::size_t>(target_rank)] + target_id;

                if (distances[other_node_id] <= new_distance) {
                    continue;
                }

                distances[other_node_id] = new_distance;
                shortest_paths_queue.emplace(target_rank, target_id, new_distance);
            }
        }

        return summarize_distances(distances);
    }
};