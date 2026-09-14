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

#include "metrics/flow/MaxFlowVariants/FlowNetwork.h"

#include <cstddef>
#include <span>

class DistributedGraph;

/**
 * @brief Maximum flow via a synchronous distributed push-relabel algorithm, following Goldberg and
 *		Tarjan (1988), "A new approach to the maximum-flow problem". Unlike the other variants this
 *		one does not gather the graph on the root rank: every rank keeps the residual capacities of
 *		the arcs that start at its own nodes together with the heights and excesses of those nodes,
 *		and only mirrors the heights of its remote neighbors.
 *		The computation proceeds in globally synchronized pulses: all nodes with excess push it over
 *		admissible arcs judged by the heights of the previous pulse, the pushed amounts travel to
 *		the receiving ranks in one collective exchange, nodes that scanned all their arcs without
 *		getting rid of their excess relabel, and finally the boundary heights are exchanged. Since
 *		all pushes of a pulse are admissible with respect to one consistent height snapshot, two
 *		nodes can never push towards each other in the same pulse, and the sequential push-relabel
 *		invariants carry over; the algorithm needs O(V^2) pulses. Periodically all heights are
 *		recomputed as exact residual distances by distributed breadth-first searches (the classic
 *		global-relabeling heuristic), which keeps the excess on shortest residual paths.
 *		Multiple sources and sinks need no super nodes here: the sources are pinned at height V and
 *		initially saturate their arcs, the sinks are pinned at height 0 and only absorb, and the
 *		flow is the total excess that arrives at the sinks, which equals the value the sequential
 *		variants compute with the super source and the super sink.
 *		Deterministic: every phase iterates the nodes and arcs in a fixed order, and the exchanged
 *		amounts are combined by commutative additions.
 */
class DistributedPushRelabelVariant {
public:
    /**
     * @brief Computes the maximum flow from the sources to the sinks on the distributed graph.
     *		Performs collective communication and must therefore be called on every MPI rank.
     * @param graph The graph whose arc weights act as capacities by absolute value
     * @param source_global_ids The distinct global ids of the sources; must be disjoint from the sinks
     * @param sink_global_ids The distinct global ids of the sinks
     * @param node_distribution The number of nodes on each MPI rank
     * @param prefix_distribution The exclusive prefix sum of node_distribution
     * @return The maximum flow value; the same on every MPI rank
     */
    [[nodiscard]] static max_flow::FlowNetwork::capacity_type compute_maximum_flow(
        const DistributedGraph& graph, std::span<const std::size_t> source_global_ids, std::span<const std::size_t> sink_global_ids,
        std::span<const global_node_id_type> node_distribution, std::span<const global_node_id_type> prefix_distribution);
};
