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

#include "mpi-wrapper/MPIReductions.h"

class InArcCounter {
public:
    /**
     * @brief Returns the total number of in arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of in arcs
     */
    [[nodiscard]] static arc_id_type count_in_arcs(const DistributedGraph& graph) {
        const auto number_local_in_arcs = graph.get_number_local_in_arcs();
        const auto number_total_in_arcs = mpiPP::MPIReductions::reduce_sum(number_local_in_arcs);

        return number_total_in_arcs;
    }

    /**
     * @brief Returns the total number of in arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of in arcs
     */
    [[nodiscard]] static arc_id_type all_count_in_arcs(const DistributedGraph& graph) {
        const auto number_local_in_arcs = graph.get_number_local_in_arcs();
        const auto number_total_in_arcs = mpiPP::MPIReductions::all_reduce_sum(number_local_in_arcs);

        return number_total_in_arcs;
    }
};

class OutArcCounter {
public:
    /**
     * @brief Returns the total number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of out arcs
     */
    [[nodiscard]] static arc_id_type count_out_arcs(const DistributedGraph& graph) {
        const auto number_local_out_arcs = graph.get_number_local_out_arcs();
        const auto number_total_out_arcs = mpiPP::MPIReductions::reduce_sum(number_local_out_arcs);

        return number_total_out_arcs;
    }

    /**
     * @brief Returns the total number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of out arcs
     */
    [[nodiscard]] static arc_id_type all_count_out_arcs(const DistributedGraph& graph) {
        const auto number_local_out_arcs = graph.get_number_local_out_arcs();
        const auto number_total_out_arcs = mpiPP::MPIReductions::all_reduce_sum(number_local_out_arcs);

        return number_total_out_arcs;
    }
};
