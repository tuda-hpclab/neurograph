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

/**
 * @brief Functions to compute the length of all arcs and to count them in
 *		a length histogram. Functions must be called on all ranks simultaneously.
 */
class ArcLengthHistogram {
public:
    /*
     * @brief Computes a histogram of the arc lengths in the graph with a fixed bin width.
     *      The arcs are weighted by their weight.
     * @param graph The graph
     * @param bin_width The width of the bins in the histogram
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<double> compute_histogram_fixed_bin_width(const DistributedGraph& graph, const double bin_width) {
        auto histogram = FixedWidthHistogram<double>{ bin_width };
        add_to_histogram(graph, histogram);

        auto result = Histogram<double>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

    /*
     * @brief Computes a histogram of the arc lengths in the graph with a fixed bin count.
     *      The arcs are weighted by their weight.
     * @param graph The graph
     * @param minimum The minimum length
     * @param maximum The maximum length
     * @param number_bins The number of bins in the histogram
     * @exception Throws an exception if any arc is longer than the maximum or shorter than the minimum
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<double> compute_histogram_fixed_bin_count(const DistributedGraph& graph, const double minimum, const double maximum,
                                                                             const std::size_t number_bins) {

        auto histogram = FixedSizeHistogram<double>{ minimum, maximum, number_bins };
        add_to_histogram(graph, histogram);

        auto result = Histogram<double>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

private:
    template <typename HistogramType>
    static void add_to_histogram(const DistributedGraph& graph, HistogramType& histogram) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            const auto& node_position = graph.get_node_position(my_rank, node_id);
            const auto out_arcs = graph.get_out_arcs(my_rank, node_id);

            for (const auto& [target_rank, target_id, weight] : out_arcs) {
                const auto& position = graph.get_node_position(target_rank, target_id);

                const auto difference = position - node_position;
                const auto distance = difference.calculate_2_norm();

                for (auto _ = weight_type{ 0 }; _ < weight; _++) {
                    histogram.add_data_point(distance);
                }
            }
        }
    }
};
