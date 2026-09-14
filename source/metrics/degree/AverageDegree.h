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
#include "metrics/counting/InArcCounter.h"
#include "metrics/counting/InWeightCounter.h"
#include "metrics/counting/OutArcCounter.h"
#include "metrics/counting/OutWeightCounter.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

class AverageDegree {
public:
    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the in degree of all in parts of the arcs, i.e., for arcs i->j, sums the in degree of i.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_in_in_degree_by_arcs(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_out(graph);
        const auto total_number_in_arcs = InArcCounter::all_count_in_arcs(graph);

        return safe_average(total_degree, total_number_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the total weight of the arcs.
     *		Checks the in degree of all in parts of the arcs, i.e., for arcs i->j, sums the in degree of i.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_in_in_degree_by_weights(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_out(graph);
        const auto total_weight_in_arcs = InWeightCounter::all_weigh_in_arcs(graph);

        return safe_average(total_degree, total_weight_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the out degree of all in parts of the arcs, i.e., for arcs i->j, sums the out degree of i.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_in_out_degree_by_arcs(const DistributedGraph& graph) {
        const auto total_degree = sum_out_times_out(graph);
        const auto total_number_in_arcs = InArcCounter::all_count_in_arcs(graph);

        return safe_average(total_degree, total_number_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the total weight of the arcs.
     *		Checks the out degree of all in parts of the arcs, i.e., for arcs i->j, sums the out degree of i.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_in_out_degree_by_weights(const DistributedGraph& graph) {
        const auto total_degree = sum_out_times_out(graph);
        const auto total_weight_in_arcs = InWeightCounter::all_weigh_in_arcs(graph);

        return safe_average(total_degree, total_weight_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the in degree of all out parts of the arcs, i.e., for arcs i->j, sums the in degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_in_degree_by_arcs(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_in(graph);
        const auto total_number_out_arcs = OutArcCounter::all_count_out_arcs(graph);

        return safe_average(total_degree, total_number_out_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the total weight of the arcs.
     *		Checks the in degree of all out parts of the arcs, i.e., for arcs i->j, sums the in degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_in_degree_by_weights(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_in(graph);
        const auto total_weight_out_arcs = OutWeightCounter::all_weigh_out_arcs(graph);

        return safe_average(total_degree, total_weight_out_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the out degree of all out parts of the arcs, i.e., for arcs i->j, sums the out degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_out_degree_by_arcs(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_out(graph);
        const auto total_number_out_arcs = OutArcCounter::all_count_out_arcs(graph);

        return safe_average(total_degree, total_number_out_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the total weight of the arcs.
     *		Checks the out degree of all out parts of the arcs, i.e., for arcs i->j, sums the out degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_out_degree_by_weights(const DistributedGraph& graph) {
        const auto total_degree = sum_in_times_out(graph);
        const auto total_weight_out_arcs = OutWeightCounter::all_weigh_out_arcs(graph);

        return safe_average(total_degree, total_weight_out_arcs);
    }

private:
    /**
     * @brief Divides the numerator by the denominator, returning 0.0 if the denominator is 0
     *		(a graph without any arcs has no average degree).
     * @param numerator The numerator
     * @param denominator The denominator
     * @return The ratio, or 0.0 if the denominator is 0
     */
    [[nodiscard]] static double safe_average(const distance_type numerator, const distance_type denominator) {
        if (denominator == 0) {
            return 0.0;
        }

        return static_cast<double>(numerator) / static_cast<double>(denominator);
    }

    /**
     * @brief Computes the globally reduced sum over all nodes of (in weight * out weight).
     * @param graph The distributed graph
     * @return The globally reduced sum on every MPI rank
     */
    [[nodiscard]] static distance_type sum_in_times_out(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto in_degree = utility::safe_cast<distance_type>(graph.get_weight_in_arcs(my_rank, node_id));
            const auto out_degree = utility::safe_cast<distance_type>(graph.get_weight_out_arcs(my_rank, node_id));
            current_degree += in_degree * out_degree;
        }

        return mpiPP::MPIReductions::all_reduce_sum(current_degree);
    }

    /**
     * @brief Computes the globally reduced sum over all nodes of (out weight * out weight).
     * @param graph The distributed graph
     * @return The globally reduced sum on every MPI rank
     */
    [[nodiscard]] static distance_type sum_out_times_out(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = utility::safe_cast<distance_type>(graph.get_weight_out_arcs(my_rank, node_id));
            current_degree += out_degree * out_degree;
        }

        return mpiPP::MPIReductions::all_reduce_sum(current_degree);
    }

    /**
     * @brief Computes the globally reduced sum over all nodes of (in weight * in weight).
     * @param graph The distributed graph
     * @return The globally reduced sum on every MPI rank
     */
    [[nodiscard]] static distance_type sum_in_times_in(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto in_degree = utility::safe_cast<distance_type>(graph.get_weight_in_arcs(my_rank, node_id));
            current_degree += in_degree * in_degree;
        }

        return mpiPP::MPIReductions::all_reduce_sum(current_degree);
    }
};
