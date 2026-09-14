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
#include "metrics/paths/DijkstraVariants/BucketQueueVariant.h"
#include "metrics/paths/DijkstraVariants/FibonacciHeapVariant.h"
#include "metrics/paths/DijkstraVariants/PriorityQueueVariant.h"
#include "metrics/paths/types/SsspResult.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

class Dijkstra {
public:
    /**
     * @brief Performs Dijkstra's algorithm on graph starting at node_id.
     *		Dispatches to the vertex queue implementation selected by queue_type; the complexity
     *		depends on that choice (see the types in metrics/paths/DijkstraVariants).
     *      Changes the behavior depending on d_type:
     *          - distance_type: Uses the absolute weight of the arcs as distances
     *          - inverse_distance_type: Uses 1/|weight| of the arcs as distances; zero-weight arcs
     *            are treated as infinitely long and never relaxed
     * @param graph The graph
     * @param node_id The starting vertex; must be owned by the calling MPI rank
     * @param prefix_distribution The prefix sum of the number of nodes on each MPI rank
     * @param distances A buffer with one entry per global node that holds the distances; only allocated once and re-used
     * @param queue_type Which vertex queue implementation to use; BucketQueue requires d_type == distance_type
     * @tparam d_type The type for the distances, can be distance_type (default) or inverse_distance_type
     * @return A quadruple of
     *		(1) the sum of all found shortest-path distances,
     *		(2) the maximum finite distance found (i.e., the eccentricity of node_id),
     *		(3) the sum of all efficiencies of the shortest paths,
     *		(4) the number of unreachable nodes
     */
    template <typename d_type = distance_type>
    [[nodiscard]] static SsspResult<d_type> compute_sssp(const DistributedGraph& graph, const node_id_type node_id,
                                                          const std::span<const global_node_id_type> prefix_distribution, std::vector<d_type>& distances,
                                                          const DijkstraQueueType queue_type = DijkstraQueueType::PriorityQueue) {
        static_assert(std::is_same_v<d_type, distance_type> || std::is_same_v<d_type, inverse_distance_type>,
                      "Dijkstra::compute_sssp only supports distance_type and inverse_distance_type");

        constexpr auto max_distance = std::numeric_limits<d_type>::max();

        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        utility::Exception::check(utility::safe_cast<std::size_t>(my_rank) < prefix_distribution.size(),
                                  "Dijkstra::compute_sssp: prefix_distribution has {} entries, which does not cover rank {}", prefix_distribution.size(), my_rank);
        utility::Exception::check(node_id < graph.get_number_local_nodes(),
                                  "Dijkstra::compute_sssp: node_id {} is out of range, rank {} owns {} nodes", node_id, my_rank, graph.get_number_local_nodes());

        std::ranges::fill(distances, max_distance);

        const auto root_id = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)] + node_id;
        utility::Exception::check(root_id < distances.size(),
                                  "Dijkstra::compute_sssp: distances buffer has {} entries, which does not cover the root's global id {}", distances.size(), root_id);
        distances[root_id] = 0;

        switch (queue_type) {
        case DijkstraQueueType::PriorityQueue:
            PriorityQueueVariant::compute_distances<d_type>(graph, node_id, prefix_distribution, distances);
            break;
        case DijkstraQueueType::FibonacciHeap:
            FibonacciHeapVariant::compute_distances<d_type>(graph, node_id, prefix_distribution, distances);
            break;
        case DijkstraQueueType::BucketQueue: {
            if constexpr (std::is_same_v<d_type, distance_type>) {
                BucketQueueVariant::compute_distances(graph, node_id, prefix_distribution, distances);
                break;
            } else {
                utility::Exception::fail("Dijkstra::compute_sssp: the bucket queue requires integer distances and cannot be used with the inverse metric");
            }
        }
        default:
            utility::Exception::fail("Dijkstra::compute_sssp: the queue type is unknown");
        }

        return summarize_distances<d_type>(distances);
    }
};
