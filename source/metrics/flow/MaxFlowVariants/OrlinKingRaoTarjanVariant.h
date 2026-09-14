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

#include <cpp-utility/Exception.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

/**
 * @brief Maximum flow in the lineage of King, Rao, and Tarjan (1994), "A faster deterministic
 *		maximum flow algorithm", and Orlin (2013), "Max flows in O(nm) time, or better", which
 *		together achieve O(V * A) across all graph densities. Both build on the push-relabel
 *		method of Goldberg and Tarjan (1988): instead of augmenting whole paths, nodes push their
 *		excess flow over single admissible arcs, ordered by height labels. King-Rao-Tarjan add
 *		dynamic trees and a derandomized game strategy for choosing the arcs, and Orlin adds a
 *		contraction scheme for abundant arcs; those additions only pay off asymptotically, so this
 *		variant implements the shared core of the lineage, the FIFO push-relabel algorithm: active
 *		nodes discharge their whole excess in first-in-first-out order (which makes the runs
 *		deterministic) and relabel to one above their lowest residual neighbor, giving O(V^3).
 */
class OrlinKingRaoTarjanVariant {
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
        const auto number_nodes = network.get_number_nodes();

        auto heights = std::vector<std::size_t>(number_nodes, 0);
        auto excesses = std::vector<max_flow::FlowNetwork::capacity_type>(number_nodes, 0);
        auto current_arc_indices = std::vector<std::size_t>(number_nodes, 0);
        auto is_active = std::vector<char>(number_nodes, 0);

        auto active_nodes = std::queue<std::size_t>{};

        // The source and the sink never become active: the source only sheds flow and the sink only collects it
        const auto activate = [&](const std::size_t node) {
            if (node != source && node != sink && is_active[node] == 0) {
                is_active[node] = 1;
                active_nodes.push(node);
            }
        };

        // The source starts at height V, so its excess can only travel towards the sink at height
        // 0, and flow that cannot reach the sink returns once nodes are lifted above the source.
        heights[source] = number_nodes;

        for (const auto arc_id : network.get_arc_ids(source)) {
            const auto amount = network.get_arc(arc_id).capacity;

            if (amount <= 0) {
                continue;
            }

            const auto target = network.get_arc(arc_id).target;

            network.push_flow(arc_id, amount);
            excesses[source] -= amount;
            excesses[target] += amount;
            activate(target);
        }

        while (!active_nodes.empty()) {
            const auto node = active_nodes.front();
            active_nodes.pop();
            is_active[node] = 0;

            // Discharge the node: push its whole excess over admissible arcs, i.e., residual arcs
            // that lead exactly one height downwards, and relabel once all arcs were scanned
            while (excesses[node] > 0) {
                const auto arc_ids = network.get_arc_ids(node);

                if (current_arc_indices[node] == arc_ids.size()) {
                    auto lowest_height = std::numeric_limits<std::size_t>::max();

                    for (const auto arc_id : arc_ids) {
                        if (network.get_arc(arc_id).capacity > 0) {
                            lowest_height = std::min(lowest_height, heights[network.get_arc(arc_id).target]);
                        }
                    }

                    // The excess arrived over some arc, so its reverse arc keeps the node connected
                    utility::Exception::check(lowest_height != std::numeric_limits<std::size_t>::max(),
                                              "OrlinKingRaoTarjanVariant::compute_maximum_flow: a node with excess has no residual arc");

                    heights[node] = lowest_height + 1;
                    current_arc_indices[node] = 0;
                    continue;
                }

                const auto arc_id = arc_ids[current_arc_indices[node]];
                const auto& arc = network.get_arc(arc_id);

                if (arc.capacity > 0 && heights[node] == heights[arc.target] + 1) {
                    const auto amount = std::min(excesses[node], arc.capacity);

                    network.push_flow(arc_id, amount);
                    excesses[node] -= amount;
                    excesses[arc.target] += amount;
                    activate(arc.target);
                } else {
                    ++current_arc_indices[node];
                }
            }
        }

        // Every intermediate node discharged completely, so all flow either reached the sink or returned
        return excesses[sink];
    }
};
