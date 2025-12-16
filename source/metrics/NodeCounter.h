#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "graph/DistributedGraph.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPICollectives.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"

#include <vector>

class NodeCounter {
public:
    /**
     * @brief Returns the total number of nodes in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of nodes
     */
    [[nodiscard]] static global_node_id_type count_nodes(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto cast = utility::save_cast<global_node_id_type>(number_local_nodes);
        const auto number_total_nodes = mpiPP::MPIReductions::reduce_sum(cast);

        return number_total_nodes;
    }

    /**
     * @brief Returns the total number of nodes in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of nodes
     */
    [[nodiscard]] static global_node_id_type all_count_nodes(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto cast = utility::save_cast<global_node_id_type>(number_local_nodes);
        const auto number_total_nodes = mpiPP::MPIReductions::all_reduce_sum(cast);

        return number_total_nodes;
    }
};

class NodeDistributionCounter {
public:
    /**
     * @brief Returns a vector that holds the number of nodes for each MPI rank.
     *		Must be called on every MPI rank, returns the result only every MPI rank.
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
     *		Must be called on every MPI rank, returns the result only every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return <return>[i] == k indicates that MPI rank i has k nodes
     */
    [[nodiscard]] static std::vector<global_node_id_type> all_count_node_distribution_global(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto cast = utility::save_cast<global_node_id_type>(number_local_nodes);
        const auto node_distribution = mpiPP::MPICollectives::all_gather(cast);

        return node_distribution;
    }
};
