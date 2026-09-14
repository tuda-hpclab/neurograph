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

#include "metrics/flow/MaxFlowVariants/FlowNetwork.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

/**
 * @brief Maximum flow after Dinitz (1970), commonly known as Dinic's algorithm: the augmentation
 *		runs in phases. Every phase layers the residual network into a breadth-first level graph
 *		and saturates it with a blocking flow, found depth first along level-increasing arcs with
 *		current-arc pointers, so every arc is scanned at most once per phase. The level of the sink
 *		grows from phase to phase, which bounds the number of phases by V and the total running
 *		time by O(V^2 * A); on unit capacities it improves to O(A * min(A^(1/2), V^(2/3))).
 */
class DinicVariant {
public:
    /**
     * @brief Computes the maximum flow from the source to the sink.
     * @param network The residual network; afterwards it holds the residual capacities of one maximum flow
     * @param source The source node id
     * @param sink The sink node id; must differ from the source
     * @return The maximum flow value
     */
    [[nodiscard]] static max_flow::FlowNetwork::capacity_type compute_maximum_flow(max_flow::FlowNetwork& network, const std::size_t source, const std::size_t sink) {
        auto total_flow = max_flow::FlowNetwork::capacity_type{ 0 };

        auto levels = std::vector<std::size_t>(network.get_number_nodes());
        auto current_arc_indices = std::vector<std::size_t>(network.get_number_nodes());
        auto queue = std::vector<std::size_t>{};

        while (true) {
            // Layer the residual network: breadth-first distances from the source
            std::ranges::fill(levels, unreachable);
            levels[source] = 0;

            queue.clear();
            queue.push_back(source);

            for (auto head = std::size_t{ 0 }; head < queue.size(); ++head) {
                const auto node = queue[head];

                for (const auto arc_id : network.get_arc_ids(node)) {
                    const auto& arc = network.get_arc(arc_id);

                    if (arc.capacity <= 0 || levels[arc.target] != unreachable) {
                        continue;
                    }

                    levels[arc.target] = levels[node] + 1;
                    queue.push_back(arc.target);
                }
            }

            if (levels[sink] == unreachable) {
                return total_flow;
            }

            total_flow += compute_blocking_flow(network, levels, current_arc_indices, source, sink);
        }
    }

private:
    static constexpr auto unreachable = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Saturates the current level graph with a blocking flow: repeatedly walks depth first
     *		along arcs that lead into the next level, augments whenever the sink is reached, and
     *		retreats over arcs that lead into exhausted nodes. The current-arc pointers survive the
     *		retreats and the augmentations, so every arc is scanned at most once per phase.
     * @param network The residual network; the capacities along the augmenting paths are updated
     * @param levels The breadth-first level per node
     * @param current_arc_indices Per node the index of the next arc to scan; only re-used as a buffer
     * @param source The source node id
     * @param sink The sink node id
     * @return The value of the blocking flow
     */
    [[nodiscard]] static max_flow::FlowNetwork::capacity_type compute_blocking_flow(max_flow::FlowNetwork& network, const std::span<const std::size_t> levels,
                                                                                    std::vector<std::size_t>& current_arc_indices,
                                                                                    const std::size_t source, const std::size_t sink) {
        auto blocking_flow = max_flow::FlowNetwork::capacity_type{ 0 };

        std::ranges::fill(current_arc_indices, std::size_t{ 0 });

        // The current path: path_arc_ids[i] leaves path_nodes[i], and path_nodes.back() is the search head
        auto path_nodes = std::vector<std::size_t>{ source };
        auto path_arc_ids = std::vector<std::size_t>{};

        while (true) {
            const auto node = path_nodes.back();

            if (node == sink) {
                // Augment along the path, then cut the path back to the first saturated arc
                auto bottleneck = std::numeric_limits<max_flow::FlowNetwork::capacity_type>::max();
                for (const auto arc_id : path_arc_ids) {
                    bottleneck = std::min(bottleneck, network.get_arc(arc_id).capacity);
                }

                for (const auto arc_id : path_arc_ids) {
                    network.push_flow(arc_id, bottleneck);
                }

                blocking_flow += bottleneck;

                auto first_saturated = std::size_t{ 0 };
                while (network.get_arc(path_arc_ids[first_saturated]).capacity > 0) {
                    ++first_saturated;
                }

                path_nodes.resize(first_saturated + 1);
                path_arc_ids.resize(first_saturated);
                continue;
            }

            const auto arc_ids = network.get_arc_ids(node);
            auto advanced = false;

            for (auto& index = current_arc_indices[node]; index < arc_ids.size(); ++index) {
                const auto arc_id = arc_ids[index];
                const auto& arc = network.get_arc(arc_id);

                if (arc.capacity > 0 && levels[arc.target] == levels[node] + 1) {
                    path_arc_ids.push_back(arc_id);
                    path_nodes.push_back(arc.target);
                    advanced = true;
                    break;
                }
            }

            if (advanced) {
                continue;
            }

            if (node == source) {
                // No admissible arc leaves the source anymore, so the flow is blocking and the phase ends
                return blocking_flow;
            }

            // Retreat: the head is exhausted, so its predecessor skips the arc that led into it
            path_nodes.pop_back();
            path_arc_ids.pop_back();
            ++current_arc_indices[path_nodes.back()];
        }
    }
};
