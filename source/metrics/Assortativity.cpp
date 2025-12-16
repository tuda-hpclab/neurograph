/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Assortativity.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/DegreeCounter.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"

#include <cmath>
#include <tuple>

std::tuple<double, double, double, double> Assortativity::compute_assortativity(const DistributedGraph& graph) {
    const auto average_in_in_degree = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out_degree = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in_degree = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out_degree = AverageDegree::all_compute_average_out_out_degree(graph);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto number_local_nodes = graph.get_number_local_nodes();

    auto local_in_in_degree_variance = 0.0;
    auto local_in_out_degree_variance = 0.0;
    auto local_out_in_degree_variance = 0.0;
    auto local_out_out_degree_variance = 0.0;

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        for (const auto& [target_rank, target_id, _] : graph.get_out_arcs(my_rank, node_id)) {
            const auto number_in_in_arcs = graph.get_weight_in_arcs(my_rank, node_id);
            const auto number_in_out_arcs = graph.get_weight_out_arcs(my_rank, node_id);
            const auto number_out_in_arcs = graph.get_weight_in_arcs(target_rank, target_id);
            const auto number_out_out_arcs = graph.get_weight_out_arcs(target_rank, target_id);

            const auto diff_in_in = static_cast<double>(number_in_in_arcs) - average_in_in_degree;
            const auto diff_in_out = static_cast<double>(number_in_out_arcs) - average_in_out_degree;
            const auto diff_out_in = static_cast<double>(number_out_in_arcs) - average_out_in_degree;
            const auto diff_out_out = static_cast<double>(number_out_out_arcs) - average_out_out_degree;

            local_in_in_degree_variance += (diff_in_in * diff_in_in);
            local_in_out_degree_variance += (diff_in_out * diff_in_out);
            local_out_in_degree_variance += (diff_out_in * diff_out_in);
            local_out_out_degree_variance += (diff_out_out * diff_out_out);
        }
    }

    const auto total_in_in_degree_variance = mpiPP::MPIReductions::all_reduce_sum(local_in_in_degree_variance);
    const auto in_in_degree_stddev = std::sqrt(total_in_in_degree_variance);

    const auto total_in_out_degree_variance = mpiPP::MPIReductions::all_reduce_sum(local_in_out_degree_variance);
    const auto in_out_degree_stddev = std::sqrt(total_in_out_degree_variance);

    const auto total_out_in_degree_variance = mpiPP::MPIReductions::all_reduce_sum(local_out_in_degree_variance);
    const auto out_in_degree_stddev = std::sqrt(total_out_in_degree_variance);

    const auto total_out_out_degree_variance = mpiPP::MPIReductions::all_reduce_sum(local_out_out_degree_variance);
    const auto out_out_degree_stddev = std::sqrt(total_out_out_degree_variance);

    auto calculate_r = [&graph, number_local_nodes, my_rank](const double std_1, const double std_2, const double avg_1, const double avg_2,
                                                             const auto& my_number_arc_getter, const auto& outer_number_arc_getter) {
        const auto product_std = std_1 * std_2;
        if (product_std == 0.0) {
            return 0.0;
        }

        auto local_r = 0.0;
        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto my_degree = my_number_arc_getter(my_rank, node_id);

            for (const auto& [target_rank, target_id, _] : graph.get_out_arcs(my_rank, node_id)) {
                const auto other_degree = outer_number_arc_getter(target_rank, target_id);
                const auto diff_1 = static_cast<double>(my_degree) - avg_1;
                const auto diff_2 = static_cast<double>(other_degree) - avg_2;

                local_r += diff_1 * diff_2;
            }
        }

        const auto total_r = mpiPP::MPIReductions::reduce_sum(local_r);
        return total_r / product_std;
    };

    const auto num_in_arc_getter = [&graph](const mpi_rank_type mpi_rank, const node_id_type node_id) { return graph.get_weight_in_arcs(mpi_rank, node_id); };
    const auto num_out_arc_getter = [&graph](const mpi_rank_type mpi_rank, const node_id_type node_id) { return graph.get_weight_out_arcs(mpi_rank, node_id); };

    const auto r_in_in = calculate_r(in_in_degree_stddev, out_in_degree_stddev, average_in_in_degree, average_in_in_degree, num_in_arc_getter, num_in_arc_getter);

    const auto r_in_out = calculate_r(in_in_degree_stddev, out_out_degree_stddev, average_in_in_degree, average_in_out_degree, num_in_arc_getter, num_out_arc_getter);

    const auto r_out_in = calculate_r(in_out_degree_stddev, out_in_degree_stddev, average_in_out_degree, average_in_in_degree, num_out_arc_getter, num_in_arc_getter);

    const auto r_out_out = calculate_r(in_out_degree_stddev, out_out_degree_stddev, average_in_out_degree, average_in_out_degree, num_out_arc_getter, num_out_arc_getter);

    return { r_in_in, r_in_out, r_out_in, r_out_out };
}
