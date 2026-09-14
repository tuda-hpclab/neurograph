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

#include <utility>

class DistributedGraph;

class Reciprocity {
public:
    /**
     * @brief Computes the reciprocity of the graph, i.e., the fraction of the arcs i->j for which the
     *		opposite arc j->i exists as well, together with the density-corrected coefficient
     *		by Garlaschelli and Loffredo (2004):
     *
     *		(1) r = A_mutual / A
     *		(2) rho = (r - a) / (1 - a), where a = A / (N * (N - 1)) is the arc density
     *
     *		While r alone grows with the density of the graph, rho relates it to the reciprocity that a
     *		random graph of the same density would show: it is positive for a graph with more mutual arcs
     *		than expected, negative for one with fewer, and zero if the two match.
     *		Self arcs are ignored entirely (they are trivially their own opposite) and so are the weights.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A * log d), where d is the maximum in-degree, neglecting MPI collectives.
     *		Needs no communication besides the collectives, because the opposite of a local out arc
     *		always shows up among the in arcs of the same local node.
     * @param graph The graph
     * @return The reciprocity from [0, 1] and the coefficient from [-1, 1]. Both are zero for a graph
     *		without arcs; the coefficient is zero for a complete graph, where it is undefined.
     */
    [[nodiscard]] static std::pair<double, double> compute_reciprocity(const DistributedGraph& graph);
};
