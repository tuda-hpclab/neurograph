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

class StronglyConnectedComponents {
public:
    /**
     * @brief Determines the strongly connected components of the graph with the selected algorithm,
     *		i.e., the maximal sets of nodes in which every node reaches every other one, and summarizes
     *		them by their sizes. Self arcs never join two nodes and are therefore ignored, as are the
     *		weights; a node without any arc forms a component of its own. All algorithms are
     *		deterministic and find the same components, so they only differ in their running time:
     *		TarjanGathered gathers the arcs on the root rank and computes the components there, while the
     *		other three keep the graph distributed (see the types in metrics/connectivity/SccVariants).
     *		Must be called on every MPI rank with the same algorithm, returns the same result on every
     *		MPI rank.
     * @param graph The graph
     * @param algorithm Which algorithm to use; Multistep is the all-round choice on a distributed graph
     * @param tail_threshold How many nodes Multistep may leave over for the root rank to finish
     *		sequentially; zero keeps every phase distributed. The other algorithms ignore it
     * @return The components, summarized by their number, their largest size, and their size distribution
     */
    [[nodiscard]] static SccResult compute_strongly_connected_components(const DistributedGraph& graph, SccAlgorithm algorithm = SccAlgorithm::Multistep,
                                                                         global_node_id_type tail_threshold = 0);
};
