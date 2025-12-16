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
#include "metrics/WeightCounter.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"

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
    [[nodiscard]] static double all_compute_average_in_in_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto in_degree = utility::save_cast<distance_type>(graph.get_weight_in_arcs(my_rank, node_id));
            const auto out_degree = utility::save_cast<distance_type>(graph.get_weight_out_arcs(my_rank, node_id));
            current_degree += in_degree * out_degree;
        }

        const auto total_number_in_arcs = InWeightCounter::all_weigh_in_arcs(graph);
        const auto total_degree = mpiPP::MPIReductions::all_reduce_sum(current_degree);

        return static_cast<double>(total_degree) / static_cast<double>(total_number_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the out degree of all in parts of the arcs, i.e., for arcs i->j, sums the out degree of i.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_in_out_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto out_degree = utility::save_cast<distance_type>(graph.get_weight_out_arcs(my_rank, node_id));
            current_degree += out_degree * out_degree;
        }

        const auto total_number_in_arcs = InWeightCounter::all_weigh_in_arcs(graph);
        const auto total_degree = mpiPP::MPIReductions::all_reduce_sum(current_degree);

        return static_cast<double>(total_degree) / static_cast<double>(total_number_in_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the in degree of all out parts of the arcs, i.e., for arcs i->j, sums the in degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_in_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto in_degree = utility::save_cast<distance_type>(graph.get_weight_in_arcs(my_rank, node_id));
            current_degree += in_degree * in_degree;
        }

        const auto total_number_out_arcs = OutWeightCounter::all_weigh_out_arcs(graph);
        const auto total_degree = mpiPP::MPIReductions::all_reduce_sum(current_degree);

        return static_cast<double>(total_degree) / static_cast<double>(total_number_out_arcs);
    }

    /**
     * @brief Returns the weighted average degree in the graph divided by the number of arcs.
     *		Checks the out degree of all out parts of the arcs, i.e., for arcs i->j, sums the out degree of j.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(V/R + R), possibly O(V/R + log R).
     * @param graph The distributed graph
     * @return A weighted average degree
     */
    [[nodiscard]] static double all_compute_average_out_out_degree(const DistributedGraph& graph) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto current_degree = distance_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto in_degree = utility::save_cast<distance_type>(graph.get_weight_in_arcs(my_rank, node_id));
            const auto out_degree = utility::save_cast<distance_type>(graph.get_weight_out_arcs(my_rank, node_id));
            current_degree += in_degree * out_degree;
        }

        const auto total_number_out_arcs = OutWeightCounter::all_weigh_out_arcs(graph);
        const auto total_degree = mpiPP::MPIReductions::all_reduce_sum(current_degree);

        return static_cast<double>(total_degree) / static_cast<double>(total_number_out_arcs);
    }
};
