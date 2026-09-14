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

class DistributedGraph;

class Modularity {
public:
    /**
     * @brief Computes the modularity of the graph: detects communities with the selected algorithm
     *		and returns the directed modularity by Leicht and Newman (2008) of the found partition
     *		(on symmetric graphs this equals the classic undirected modularity). All algorithms are
     *		deterministic, so repeated runs return the same value; the weights enter by absolute value.
     *		The community-detection algorithms are inherently sequential, so the arcs are gathered
     *		on the root rank, the detection runs there, and the result is broadcast afterwards.
     *		Must be called on every MPI rank, returns the same value on every MPI rank.
     * @param graph The graph
     * @param algorithm Which community-detection algorithm to use. Only Louvain and Leiden maximize
     *		the modularity itself, so the returned value is usually lower for the other ones.
     * @return The modularity of the detected community partition; 0.0 for a graph without arcs
     */
    [[nodiscard]] static double compute_modularity(const DistributedGraph& graph, ModularityAlgorithm algorithm = ModularityAlgorithm::Louvain);
};
