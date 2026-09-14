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

/**
 * @brief Shared building blocks of the maximum-flow variants (see the sibling headers in
 *		metrics/flow/MaxFlowVariants). All variants work on the residual network (see max_flow::FlowNetwork)
 *		and scan arcs in insertion order, so every variant is deterministic and repeated runs
 *		behave identically.
 */
class MaxFlowCommon {
public:
    /** The parent-arc id that marks a node as not yet reached by a path search. */
    static constexpr auto unreached = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Walks the parent arcs from the sink back to the source, determines the bottleneck
     *		capacity of that augmenting path, and pushes the bottleneck over every arc of the path.
     * @param network The residual network; the capacities along the path are updated
     * @param parent_arc_ids Per node the id of the arc over which the path search reached it
     * @param source The node at which the path starts
     * @param sink The node at which the path ends; must have been reached from the source
     * @return The bottleneck flow that was pushed, always positive
     */
    [[nodiscard]] static max_flow::FlowNetwork::capacity_type augment_along_parent_arcs(max_flow::FlowNetwork& network,
                                                                                        const std::span<const std::size_t> parent_arc_ids,
                                                                                        const std::size_t source, const std::size_t sink) {
        // The tail of an arc is the target of its reverse arc, so the path can be walked backwards.
        const auto get_arc_tail = [&network](const std::size_t arc_id) {
            return network.get_arc(network.get_arc(arc_id).reverse_arc_id).target;
        };

        auto bottleneck = std::numeric_limits<max_flow::FlowNetwork::capacity_type>::max();

        for (auto node = sink; node != source; node = get_arc_tail(parent_arc_ids[node])) {
            bottleneck = std::min(bottleneck, network.get_arc(parent_arc_ids[node]).capacity);
        }

        for (auto node = sink; node != source; node = get_arc_tail(parent_arc_ids[node])) {
            network.push_flow(parent_arc_ids[node], bottleneck);
        }

        return bottleneck;
    }
};
