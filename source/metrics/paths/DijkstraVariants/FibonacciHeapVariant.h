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
#include <cpp-utility/data-structure/FibonacciHeap.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

/**
 * @brief Dijkstra vertex queue variant that uses a Fibonacci heap with decrease-key:
 *		every node is inserted at most once and improved in place, giving the textbook
 *		O(A + V log V) bound (in contrast to the lazy-deletion variants), neglecting MPI communication.
 */
class FibonacciHeapVariant {
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
        const auto root_id = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)] + root_node_id;

        using heap_type = utility::FibonacciHeap<d_type, NodeIdentifier>;

        auto heap = heap_type{};

        // One handle per global node; nullptr means the node has never been inserted. Handles of
        // extracted (settled) nodes become stale but are never used again: a settled node's distance
        // is final, so the relaxation check below always rejects it before the decrease_key branch.
        auto handles = std::vector<typename heap_type::Node*>(distances.size(), nullptr);
        handles[root_id] = heap.push(d_type{ 0 }, NodeIdentifier{ my_rank, root_node_id });

        while (!heap.empty()) {
            const auto [current_distance, current_node] = heap.extract_min();

            const auto out_arcs = graph.get_out_arcs(current_node.owning_mpi_rank, current_node.node_id);

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
                if (handles[other_node_id] == nullptr) {
                    handles[other_node_id] = heap.push(new_distance, NodeIdentifier{ target_rank, target_id });
                } else {
                    heap.decrease_key(handles[other_node_id], new_distance);
                }
            }
        }
    }
};
