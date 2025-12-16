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
#include "utility/Histogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <vector>

class DegreeHistogram {
public:
    /**
     * @brief Computes a histogram of the in degrees of the nodes in the graph with a fixed bin width.
     * @param graph The graph
     * @param bin_width The width of the bins in the histogram
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<arc_id_type> compute_in_degree_fixed_bin_width(const DistributedGraph& graph, const arc_id_type bin_width) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = FixedWidthHistogram<arc_id_type>(bin_width);

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto in_degree = graph.get_number_in_arcs(my_rank, local_node);
            histogram.add_data_point(in_degree);
        }

        auto result = Histogram<arc_id_type>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

    /**
     * @brief Computes a histogram of the in degrees of the nodes in the graph with a fixed bin count.
     * @param graph The graph
     * @param minimum The minimum length
     * @param maximum The maximum length
     * @param number_bins The number of bins in the histogram
     * @exception Throws an exception if any arc is longer than the maximum or shorter than the minimum
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<arc_id_type> compute_in_degree_fixed_bin_count(const DistributedGraph& graph, const arc_id_type minimum,
                                                                                  const arc_id_type maximum, const std::size_t number_bins) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = FixedSizeHistogram<arc_id_type>{ minimum, maximum, number_bins };

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto in_degree = graph.get_number_in_arcs(my_rank, local_node);
            histogram.add_data_point(in_degree);
        }

        auto result = Histogram<arc_id_type>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

    /**
     * @brief Computes a histogram of the out degrees of the nodes in the graph with a fixed bin width.
     * @param graph The graph
     * @param bin_width The width of the bins in the histogram
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<arc_id_type> compute_out_degree_fixed_bin_width(const DistributedGraph& graph, const arc_id_type bin_width) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = FixedWidthHistogram<arc_id_type>(bin_width);

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, local_node);
            histogram.add_data_point(out_degree);
        }

        auto result = Histogram<arc_id_type>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

    /**
     * @brief Computes a histogram of the out degrees of the nodes in the graph with a fixed bin count.
     * @param graph The graph
     * @param minimum The minimum length
     * @param maximum The maximum length
     * @param number_bins The number of bins in the histogram
     * @exception Throws an exception if any arc is longer than the maximum or shorter than the minimum
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<arc_id_type> compute_out_degree_fixed_bin_count(const DistributedGraph& graph, const arc_id_type minimum,
                                                                                   const arc_id_type maximum, const std::size_t number_bins) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = FixedSizeHistogram<arc_id_type>{ minimum, maximum, number_bins };

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, local_node);
            histogram.add_data_point(out_degree);
        }

        auto result = Histogram<arc_id_type>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }
};
