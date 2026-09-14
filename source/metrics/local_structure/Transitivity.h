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

class Transitivity {
public:
    /**
     * @brief Computes the transitivity of the graph, i.e., the global clustering coefficient:
     *
     *		T = (closed triangles of all nodes) / (potential triangles of all nodes)
     *
     *		Unlike the average clustering coefficient (Clustering), which averages the ratio of every
     *		node and therefore weighs a node of degree two as much as a hub, the transitivity divides
     *		the two sums and therefore weighs every node by the number of triangles it can be part of.
     *		Both counting variants of the clustering coefficient are reported:
     *		(1) the triangles of the form i->j->k->i for three pairwise distinct nodes
     *		(2) the triangles of the form i->j->k<-i for three pairwise distinct nodes
     *		Arcs from a node to itself are ignored, and so are the weights. On a symmetric graph, both
     *		variants are the classic transitivity 3 * (number of triangles) / (number of triples).
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A * d^2) in the worst case, where d is the maximum degree
     *		(each arc is compared against the neighborhoods of its endpoints), neglecting MPI collectives.
     * @param graph The graph
     * @return The transitivity of the two variants in the given order; each is from [0, 1] and zero
     *		for a graph in which no node has a potential triangle
     */
    [[nodiscard]] static std::pair<double, double> compute_transitivity(const DistributedGraph& graph);
};
