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

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <limits>

/**
 * @brief Functions to compute the minimum and maximum out degrees of the nodes.
 *		Functions must be called on all ranks simultaneously.
 */
class OutDegree {
public:
    /**
     * @brief Returns the minimum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return The minimum
     */
    [[nodiscard]] static arc_id_type compute_minimum_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_min = std::numeric_limits<arc_id_type>::max();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);
            current_min = std::min(current_min, out_degree);
        }

        const auto total_min = mpiPP::MPIReductions::reduce_min(current_min);
        return total_min;
    }

    /**
     * @brief Returns the maximum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return The maximum
     */
    [[nodiscard]] static arc_id_type compute_maximum_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_max = std::numeric_limits<arc_id_type>::min();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);
            current_max = std::max(current_max, out_degree);
        }

        const auto total_max = mpiPP::MPIReductions::reduce_max(current_max);
        return total_max;
    }

    /**
     * @brief Returns the minimum and maximum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A tuple of (1) the minimum and (2) the maximum
     */
    [[nodiscard]] static MinMax compute_extreme_degrees(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_min = std::numeric_limits<arc_id_type>::max();
        auto current_max = std::numeric_limits<arc_id_type>::min();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);

            current_min = std::min(current_min, out_degree);
            current_max = std::max(current_max, out_degree);
        }

        const auto total_min = mpiPP::MPIReductions::reduce_min(current_min);
        const auto total_max = mpiPP::MPIReductions::reduce_max(current_max);

        return { total_min, total_max };
    }

    /**
     * @brief Returns the minimum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return The minimum
     */
    [[nodiscard]] static arc_id_type all_compute_minimum_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_min = std::numeric_limits<arc_id_type>::max();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);
            current_min = std::min(current_min, out_degree);
        }

        const auto total_min = mpiPP::MPIReductions::all_reduce_min(current_min);
        return total_min;
    }

    /**
     * @brief Returns the maximum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return The maximum
     */
    [[nodiscard]] static arc_id_type all_compute_maximum_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_max = std::numeric_limits<arc_id_type>::min();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);
            current_max = std::max(current_max, out_degree);
        }

        const auto total_max = mpiPP::MPIReductions::all_reduce_max(current_max);
        return total_max;
    }

    /**
     * @brief Returns the minimum and maximum number of out arcs in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A tuple of (1) the minimum and (2) the maximum
     */
    [[nodiscard]] static MinMax all_compute_extreme_degrees(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_min = std::numeric_limits<arc_id_type>::max();
        auto current_max = std::numeric_limits<arc_id_type>::min();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, node_id);

            current_min = std::min(current_min, out_degree);
            current_max = std::max(current_max, out_degree);
        }

        const auto total_min = mpiPP::MPIReductions::all_reduce_min(current_min);
        const auto total_max = mpiPP::MPIReductions::all_reduce_max(current_max);

        return { total_min, total_max };
    }
};
