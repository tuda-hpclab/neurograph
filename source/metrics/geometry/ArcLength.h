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
#include "metrics/counting/OutWeightCounter.h"
#include "utility/Vec3.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

/**
 * @brief Functions to compute the average, minimum and maximum length of all arcs.
 *		Functions must be called on all ranks simultaneously.
 */
class ArcLength {
public:
    /**
     * @brief Computes the average arc length in the graph. Arcs are weighted by their weight.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The average arc length, or 0.0 if the graph has no arcs
     */
    [[nodiscard]] static double compute_average_arc_length(const DistributedGraph& graph) {
        const auto accumulated_distance = compute_local_sum_arc_lengths(graph);

        const auto total_distance = mpiPP::MPIReductions::reduce_sum(accumulated_distance);
        const auto total_arc_weight = OutWeightCounter::weigh_out_arcs(graph);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return 0.0;
        }

        if (total_arc_weight == 0) {
            return 0.0;
        }

        const auto average_distance = total_distance / static_cast<double>(total_arc_weight);

        return average_distance;
    }

    /**
     * @brief Computes the maximum arc length in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The maximum arc length
     */
    [[nodiscard]] static double compute_maximum_arc_length(const DistributedGraph& graph) {
        const auto maximum_distance = compute_local_max_arc_lengths(graph);
        const auto global_maximum_distance = mpiPP::MPIReductions::reduce_max(maximum_distance);
        return global_maximum_distance;
    }

    /**
     * @brief Computes the minimum arc length in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The minimum arc length
     */
    [[nodiscard]] static double compute_minimum_arc_length(const DistributedGraph& graph) {
        const auto minimum_distance = compute_local_min_arc_lengths(graph);
        const auto global_minimum_distance = mpiPP::MPIReductions::reduce_min(minimum_distance);
        return global_minimum_distance;
    }

    /**
     * @brief Computes the average arc length in the graph on every MPI rank. Arcs are weighted by their weight.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The average arc length, or 0.0 if the graph has no arcs
     */
    [[nodiscard]] static double all_compute_average_arc_length(const DistributedGraph& graph) {
        const auto accumulated_distance = compute_local_sum_arc_lengths(graph);

        const auto total_distance = mpiPP::MPIReductions::all_reduce_sum(accumulated_distance);
        const auto total_arc_weight = OutWeightCounter::all_weigh_out_arcs(graph);

        if (total_arc_weight == 0) {
            return 0.0;
        }

        const auto average_distance = total_distance / static_cast<double>(total_arc_weight);

        return average_distance;
    }

    /**
     * @brief Computes the maximum arc length in the graph on every MPI rank.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The maximum arc length
     */
    [[nodiscard]] static double all_compute_maximum_arc_length(const DistributedGraph& graph) {
        const auto maximum_distance = compute_local_max_arc_lengths(graph);
        const auto global_maximum_distance = mpiPP::MPIReductions::all_reduce_max(maximum_distance);
        return global_maximum_distance;
    }

    /**
     * @brief Computes the minimum arc length in the graph on every MPI rank.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The graph
     * @return The minimum arc length
     */
    [[nodiscard]] static double all_compute_minimum_arc_length(const DistributedGraph& graph) {
        const auto minimum_distance = compute_local_min_arc_lengths(graph);
        const auto global_minimum_distance = mpiPP::MPIReductions::all_reduce_min(minimum_distance);
        return global_minimum_distance;
    }

private:
    [[nodiscard]] static double compute_local_sum_arc_lengths(const DistributedGraph& graph) {
        const auto sum_arc_lengths = [](const DistributedGraph& _graph, const node_id_type node_id) {
            const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

            auto accumulated_distance = 0.0;
            const auto& node_position = _graph.get_node_position(my_rank, node_id);
            const auto out_arcs = _graph.get_out_arcs(my_rank, node_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto& position = _graph.get_node_position(target_rank, target_id);

                const auto difference = position - node_position;
                const auto distance = difference.calculate_2_norm();

                accumulated_distance += (distance * static_cast<double>(weight));
            }

            return accumulated_distance;
        };

        const auto number_local_nodes = graph.get_number_local_nodes();

        auto accumulated_distance = 0.0;

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto distance = sum_arc_lengths(graph, node_id);
            accumulated_distance += distance;
        }

        return accumulated_distance;
    }

    [[nodiscard]] static double compute_local_max_arc_lengths(const DistributedGraph& graph) {
        const auto max_arc_lengths = [](const DistributedGraph& _graph, const node_id_type node_id) {
            const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

            auto maximum_distance = 0.0;
            const auto& node_position = _graph.get_node_position(my_rank, node_id);
            const auto out_arcs = _graph.get_out_arcs(my_rank, node_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto& position = _graph.get_node_position(target_rank, target_id);

                const auto difference = position - node_position;
                const auto distance = difference.calculate_2_norm();

                maximum_distance = std::max(maximum_distance, distance);
            }

            return maximum_distance;
        };

        const auto number_local_nodes = graph.get_number_local_nodes();

        auto maximum_distance = 0.0;

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto distance = max_arc_lengths(graph, node_id);
            maximum_distance = std::max(maximum_distance, distance);
        }

        return maximum_distance;
    }

    [[nodiscard]] static double compute_local_min_arc_lengths(const DistributedGraph& graph) {
        const auto min_arc_lengths = [](const DistributedGraph& _graph, const node_id_type node_id) {
            const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

            auto minimum_distance = std::numeric_limits<double>::max();
            const auto& node_position = _graph.get_node_position(my_rank, node_id);
            const auto out_arcs = _graph.get_out_arcs(my_rank, node_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto& position = _graph.get_node_position(target_rank, target_id);

                const auto difference = position - node_position;
                const auto distance = difference.calculate_2_norm();

                minimum_distance = std::min(minimum_distance, distance);
            }

            return minimum_distance;
        };

        const auto number_local_nodes = graph.get_number_local_nodes();

        auto minimum_distance = std::numeric_limits<double>::max();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto distance = min_arc_lengths(graph, node_id);
            minimum_distance = std::min(minimum_distance, distance);
        }

        return minimum_distance;
    }
};
