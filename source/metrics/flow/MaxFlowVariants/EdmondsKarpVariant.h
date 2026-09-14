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
 * @brief Maximum flow after Edmonds and Karp (1972), "Theoretical improvements in algorithmic
 *		efficiency for network flow problems": the Ford-Fulkerson method, but every augmenting
 *		path is found breadth first and is therefore a shortest residual path. The shortest-path
 *		distance from the source never decreases over the augmentations, which bounds their number
 *		by O(V * A) and the total running time by O(V * A^2), independently of the capacities.
 */
class EdmondsKarpVariant {
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
        auto queue = std::vector<std::size_t>{};

        while (true) {
            std::ranges::fill(parent_arc_ids, MaxFlowCommon::unreached);

            queue.clear();
            queue.push_back(source);

            auto reached_sink = false;

            for (auto head = std::size_t{ 0 }; head < queue.size() && !reached_sink; ++head) {
                const auto node = queue[head];

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

                    queue.push_back(arc.target);
                }
            }

            if (!reached_sink) {
                return total_flow;
            }

            total_flow += MaxFlowCommon::augment_along_parent_arcs(network, parent_arc_ids, source, sink);
        }
    }
};
