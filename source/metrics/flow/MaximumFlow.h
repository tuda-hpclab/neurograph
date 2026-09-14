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

#include <span>

class DistributedGraph;

class MaximumFlow {
public:
    /**
     * @brief Computes the maximum flow from the sources to the sinks with the selected algorithm.
     *		The capacity of an arc is the absolute value of its weight; self loops carry no flow
     *		and are ignored. Multiple sources and sinks are joined through a super source and a
     *		super sink whose connecting arcs are effectively unbounded; duplicate entries within
     *		the sources or within the sinks count once. All algorithms are deterministic and
     *		compute the same value. The augmenting computations are inherently sequential, so those
     *		variants gather the arcs on the root rank, compute the flow there, and broadcast the
     *		result afterwards; DistributedPushRelabel instead keeps the graph distributed and
     *		computes the flow with all ranks. Must be called on every MPI rank with the same
     *		sources, sinks, and algorithm; returns the same value on every MPI rank.
     * @param graph The graph
     * @param sources The source nodes; must not be empty, must exist, and must be disjoint from the sinks
     * @param sinks The sink nodes; must not be empty, must exist, and must be disjoint from the sources
     * @param algorithm Which maximum-flow algorithm to use; Dinic is the strongly polynomial all-round choice
     * @return The maximum flow value
     */
    [[nodiscard]] static flow_type compute_maximum_flow(const DistributedGraph& graph, std::span<const NodeIdentifier> sources,
                                                        std::span<const NodeIdentifier> sinks, MaxFlowAlgorithm algorithm = MaxFlowAlgorithm::Dinic);
};
