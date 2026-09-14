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

#include <tuple>

class DistributedGraph;

class Assortativity {
public:
    /**
     * @brief Computes the assortativity for all nodes (Pearson Correlation Coefficient),
     *		i.e., the correlation between the degree of a pre-arc node and a post-arc node.
     *
     *		Using all arcs i->j, computes the assortativity between
     *		(1) indeg(i) and indeg(j)
     *		(2) indeg(i) and outdeg(j)
     *		(3) outdeg(i) and indeg(j)
     *		(4) outdeg(i) and outdeg(j)
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A), neglecting MPI collectives.
     * @param graph The graph
     * @return The coefficients in the given order; each is from [-1, 1]
     */
    [[nodiscard]] static std::tuple<double, double, double, double> compute_assortativity(const DistributedGraph& graph);
};
