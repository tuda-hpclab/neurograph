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

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

class OutWeightCounter {
public:
    /**
     * @brief Returns the total weight of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *      Complexity is O(R + |Nodes| / R), possibly O(log R + |Nodes| / R).
     * @param graph The distributed graph
     * @return The total weight of out arcs
     */
    [[nodiscard]] static distance_type weigh_out_arcs(const DistributedGraph& graph) {
        const auto local_out_weight = weigh_out_arcs_priv(graph);
        const auto total_out_weight = mpiPP::MPIReductions::reduce_sum(local_out_weight);

        return total_out_weight;
    }

    /**
     * @brief Returns the total weight of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *      Complexity is O(R + |Nodes| / R), possibly O(log R + |Nodes| / R).
     * @param graph The distributed graph
     * @return The total weight of out arcs
     */
    [[nodiscard]] static distance_type all_weigh_out_arcs(const DistributedGraph& graph) {
        const auto local_out_weight = weigh_out_arcs_priv(graph);
        const auto total_out_weight = mpiPP::MPIReductions::all_reduce_sum(local_out_weight);

        return total_out_weight;
    }

private:
    [[nodiscard]] static distance_type weigh_out_arcs_priv(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto local_weight = distance_type{ 0 };
        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
            const auto weight = graph.get_weight_out_arcs(my_rank, node_id);
            const auto cast = utility::safe_cast<distance_type>(weight);
            local_weight += cast;
        }

        return local_weight;
    }
};
