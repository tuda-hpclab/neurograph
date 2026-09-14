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

#include "graph/DistributedGraph.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/collectives/MPIAllGather.h>

#include <vector>

class NodeDistributionCounter {
public:
    /**
     * @brief Returns a vector that holds the number of nodes for each MPI rank.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return <return>[i] == k indicates that MPI rank i has k nodes
     */
    [[nodiscard]] static std::vector<node_id_type> all_count_node_distribution(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto node_distribution = mpiPP::MPICollectives::all_gather(number_local_nodes);

        return node_distribution;
    }

    /**
     * @brief Returns a vector that holds the number of nodes for each MPI rank; returns the values in a larger value type.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return <return>[i] == k indicates that MPI rank i has k nodes
     */
    [[nodiscard]] static std::vector<global_node_id_type> all_count_node_distribution_global(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto cast = utility::safe_cast<global_node_id_type>(number_local_nodes);
        const auto node_distribution = mpiPP::MPICollectives::all_gather(cast);

        return node_distribution;
    }
};
