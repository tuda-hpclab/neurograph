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

#include <mpi-wrapper/reductions/MPIReductions.h>

class InArcCounter {
public:
    /**
     * @brief Returns the total number of in arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of in arcs
     */
    [[nodiscard]] static distance_type count_in_arcs(const DistributedGraph& graph) {
        const auto number_local_in_arcs = graph.get_number_local_in_arcs();
        const auto cast = utility::safe_cast<distance_type>(number_local_in_arcs);
        const auto number_total_in_arcs = mpiPP::MPIReductions::reduce_sum(cast);

        return number_total_in_arcs;
    }

    /**
     * @brief Returns the total number of in arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(R), possibly O(log R).
     * @param graph The distributed graph
     * @return The total number of in arcs
     */
    [[nodiscard]] static distance_type all_count_in_arcs(const DistributedGraph& graph) {
        const auto number_local_in_arcs = graph.get_number_local_in_arcs();
        const auto cast = utility::safe_cast<distance_type>(number_local_in_arcs);
        const auto number_total_in_arcs = mpiPP::MPIReductions::all_reduce_sum(cast);

        return number_total_in_arcs;
    }
};
