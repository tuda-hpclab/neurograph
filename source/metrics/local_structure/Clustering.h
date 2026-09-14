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
#include "metrics/counting/NodeCounter.h"
#include "metrics/local_structure/TriangleCounting.h"

#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <utility>

class Clustering {
public:
    /**
     * @brief Computes the average clustering coefficient for all nodes,
     *		i.e., how many triangles exist divided by the potential number of triangles.
     *		The triangles are of the form i->j->k->i for three pairwise distinct nodes
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A * d^2) in the worst case, where d is the maximum degree
     *		(each arc is compared against the neighborhoods of its endpoints), neglecting MPI collectives.
     * @param graph The graph
     * @return The ratio of existing triangles
     */
    [[nodiscard]] static double compute_average_clustering_coefficient(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto status = mpiPP::MPIProgress{ number_local_nodes, "Clustering Coefficient" };

        auto sum_clustering_coefficient_locally = 0.0;

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
            const auto [closed_triangles, all_triangles] = TriangleCounting::count_triangles_node(graph, node_id);

            if (all_triangles > 0ULL) {
                const auto clustering_coefficient_node = static_cast<double>(closed_triangles) / static_cast<double>(all_triangles);
                sum_clustering_coefficient_locally += clustering_coefficient_node;
            }

            status.report(node_id);
        }

        status.finish();

        const auto sum_clustering_coefficient = mpiPP::MPIReductions::reduce_sum(sum_clustering_coefficient_locally);
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);
        const auto average_clustering_coefficient = sum_clustering_coefficient / static_cast<double>(total_number_nodes);

        return average_clustering_coefficient;
    }

    /**
     * @brief Computes the average clustering coefficient for all nodes,
     *		i.e., how many triangles exist divided by the potential number of triangles.
     *		The triangles are of the form i->j->k<-i for three pairwise distinct nodes
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A * d^2) in the worst case, where d is the maximum degree
     *		(each arc is compared against the neighborhoods of its endpoints), neglecting MPI collectives.
     * @param graph The graph
     * @return The ratio of existing triangles
     */
    [[nodiscard]] static double compute_average_clustering_coefficient_2(const DistributedGraph& graph) {
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto status = mpiPP::MPIProgress{ number_local_nodes, "Clustering Coefficient" };

        auto sum_clustering_coefficient_locally = 0.0;

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
            const auto [closed_triangles, all_triangles] = TriangleCounting::count_triangles_node_2(graph, node_id);

            if (all_triangles > 0ULL) {
                const auto clustering_coefficient_node = static_cast<double>(closed_triangles) / static_cast<double>(all_triangles);
                sum_clustering_coefficient_locally += clustering_coefficient_node;
            }

            status.report(node_id);
        }

        status.finish();

        const auto sum_clustering_coefficient = mpiPP::MPIReductions::reduce_sum(sum_clustering_coefficient_locally);
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);
        const auto average_clustering_coefficient = sum_clustering_coefficient / static_cast<double>(total_number_nodes);

        return average_clustering_coefficient;
    }
};
