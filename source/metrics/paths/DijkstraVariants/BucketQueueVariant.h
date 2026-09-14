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

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/data-structure/BucketQueue.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <span>
#include <vector>

/**
 * @brief Dijkstra vertex queue variant that uses a monotone integer bucket queue (Dial's algorithm)
 *		with lazy deletion. Only valid for the integer distance_type metric. Complexity is
 *		O(A + D) where D is the largest finite distance from the root, neglecting MPI communication,
 *		so this variant is best suited for small integer weights.
 */
class BucketQueueVariant {
public:
    /**
     * @brief Runs the Dijkstra main loop starting at the calling rank's root_node_id.
     * @param graph The graph
     * @param root_node_id The starting vertex, owned by the calling MPI rank
     * @param prefix_distribution The prefix sum of the number of nodes on each MPI rank
     * @param distances One entry per global node; pre-filled with the maximum value and 0 at the root
     */
    static void compute_distances(const DistributedGraph& graph, const node_id_type root_node_id,
                                  const std::span<const global_node_id_type> prefix_distribution, std::vector<distance_type>& distances) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto queue = utility::BucketQueue<distance_type, NodeIdentifier>{};
        queue.push(distance_type{ 0 }, NodeIdentifier{ my_rank, root_node_id });

        while (!queue.empty()) {
            const auto [current_distance, current_node] = queue.pop();

            const auto current_node_id = prefix_distribution[utility::safe_cast<std::size_t>(current_node.owning_mpi_rank)] + current_node.node_id;
            if (distances[current_node_id] < current_distance) {
                // Stale queue entry: a shorter distance to this node was found after it had been enqueued.
                continue;
            }

            const auto out_arcs = graph.get_out_arcs(current_node.owning_mpi_rank, current_node.node_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto arc_distance = DijkstraCommon::transform_weight<distance_type>(weight);

                const auto new_distance = current_distance + arc_distance;

                const auto other_node_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;

                if (distances[other_node_id] <= new_distance) {
                    continue;
                }

                distances[other_node_id] = new_distance;
                queue.push(new_distance, NodeIdentifier{ target_rank, target_id });
            }
        }
    }
};
