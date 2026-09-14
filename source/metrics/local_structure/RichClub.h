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

class RichClub {
public:
    /**
     * @brief Computes the rich-club coefficients of the graph after Zhou and Mondragon (2004),
     *		"The rich-club phenomenon in the internet topology": for every degree threshold k, how
     *		densely the nodes above that threshold are connected among themselves,
     *
     *		phi(k) = A_>k / (N_>k * (N_>k - 1)),
     *
     *		where N_>k is the number of nodes with a degree larger than k (the club of the threshold)
     *		and A_>k is the number of arcs that run from one of them to a different one. The
     *		denominator counts the ordered pairs of distinct club members, so phi(k) is the arc
     *		density of the subgraph induced by the club and lies in [0, 1]. A curve that rises with k
     *		means that the hubs of the graph are more densely connected among each other than the
     *		graph is on average, which is what makes them a rich club; a falling curve means the
     *		opposite, that the hubs avoid each other. Note that phi is only compared against itself
     *		here: since a node of a high degree has many arcs to place, phi rises even in a random
     *		graph, so judging whether a club is rich beyond that needs a null model of the same degree
     *		sequence, which this metric does not compute.
     *		The degree of a node is its number of in arcs plus its number of out arcs, i.e., an arc
     *		from a node to itself counts twice, once in each direction, exactly as the degree of a
     *		directed graph is defined in networkx; --remove-autapses drops those arcs beforehand.
     *		Such an arc is never counted in A_>k though, because it runs inside one node instead of
     *		between two distinct members. The weights play no role at all.
     *		On a symmetric graph the coefficient is the classic undirected one of Zhou and Mondragon
     *		at half the threshold, because both the degrees and the arc counts double there.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A), neglecting MPI collectives, and the memory per rank is O(n + d), where n
     *		is that rank's number of nodes and d the maximum degree of the graph. Needs two remote
     *		accesses per arc that leaves the rank, one per arc direction of its target, to learn that
     *		node's degree.
     * @param graph The graph
     * @return The coefficient, the club size, and the number of club arcs per degree threshold,
     *		starting at the threshold zero. The largest reported threshold is the last one whose club
     *		still holds at least two nodes, because a smaller club has no pair of distinct nodes and
     *		therefore no density; the result is empty for a graph in which fewer than two nodes have
     *		an arc at all.
     */
    [[nodiscard]] static RichClubResult compute_rich_club_coefficients(const DistributedGraph& graph);
};
