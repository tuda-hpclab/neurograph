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
#include "metrics/flow/MaxFlowVariants/MaxFlowCommon.h"

#include <algorithm>
#include <cstddef>
#include <vector>

/**
 * @brief Maximum flow after Ford and Fulkerson (1956), "Maximal flow through a network":
 *		repeatedly find any augmenting path in the residual network and push its bottleneck flow,
 *		until no residual path from the source to the sink remains. This variant searches the path
 *		depth first with an explicit stack, deterministically in arc-insertion order. Every
 *		augmentation pushes at least one unit, so the running time is O(A * F) where F is the
 *		maximum flow value; the arbitrary path choice makes the bound depend on the capacities,
 *		which Edmonds-Karp and Dinic improve on.
 */
class FordFulkersonVariant {
public:
    /**
     * @brief Computes the maximum flow from the source to the sink.
     * @param network The residual network; afterwards it holds the residual capacities of one maximum flow
     * @param source The source node id
     * @param sink The sink node id; must differ from the source
     * @return The maximum flow value
     */
    [[nodiscard]] static max_flow::FlowNetwork::capacity_type compute_maximum_flow(max_flow::FlowNetwork& network, const std::size_t source,
                                                                                   const std::size_t sink) {
        auto total_flow = max_flow::FlowNetwork::capacity_type{ 0 };

        auto parent_arc_ids = std::vector<std::size_t>(network.get_number_nodes());
        auto stack = std::vector<std::size_t>{};

        while (true) {
            std::ranges::fill(parent_arc_ids, MaxFlowCommon::unreached);

            stack.clear();
            stack.push_back(source);

            auto reached_sink = false;

            while (!stack.empty() && !reached_sink) {
                const auto node = stack.back();
                stack.pop_back();

                for (const auto arc_id : network.get_arc_ids(node)) {
                    const auto& arc = network.get_arc(arc_id);

                    // The parent arcs double as the visited marks; the source never gets one.
                    if (arc.capacity <= 0 || arc.target == source || parent_arc_ids[arc.target] != MaxFlowCommon::unreached) {
                        continue;
                    }

                    parent_arc_ids[arc.target] = arc_id;

                    if (arc.target == sink) {
                        reached_sink = true;
                        break;
                    }

                    stack.push_back(arc.target);
                }
            }

            if (!reached_sink) {
                return total_flow;
            }

            total_flow += MaxFlowCommon::augment_along_parent_arcs(network, parent_arc_ids, source, sink);
        }
    }
};
