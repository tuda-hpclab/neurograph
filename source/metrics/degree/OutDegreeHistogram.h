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

#include <cpp-utility/data-structure/Histogram.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIAdvancedReductions.h>

#include <vector>

/**
 * @brief Functions to compute histograms of the out degrees of the nodes.
 *		Functions must be called on all ranks simultaneously.
 */
class OutDegreeHistogram {
public:
    /**
     * @brief Computes a histogram of the out degrees of the nodes in the graph with a fixed bin width.
     * @param graph The graph
     * @param bin_width The width of the bins in the histogram
     * @return The globally reduced histogram on every MPI rank
     */
    [[nodiscard]] static mpiPP::MPIAdvancedReductions::HistogramReduction<arc_id_type> compute_fixed_bin_width(const DistributedGraph& graph,
                                                                                                                  const arc_id_type bin_width) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = utility::FixedWidthHistogram<arc_id_type>(bin_width);

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, local_node);
            histogram.add_data_point(out_degree);
        }

        return mpiPP::MPIAdvancedReductions::reduce_histogram(utility::Histogram<arc_id_type>{ std::move(histogram) });
    }

    /**
     * @brief Computes a histogram of the out degrees of the nodes in the graph with a fixed bin count.
     * @param graph The graph
     * @param minimum The minimum degree
     * @param maximum The maximum degree
     * @param number_bins The number of bins in the histogram
     * @exception Throws an exception if any node's degree is larger than the maximum or smaller than the minimum
     * @return The globally reduced histogram on every MPI rank
     */
    [[nodiscard]] static mpiPP::MPIAdvancedReductions::HistogramReduction<arc_id_type> compute_fixed_bin_count(const DistributedGraph& graph,
                                                                                                                  const arc_id_type minimum, const arc_id_type maximum,
                                                                                                                  const std::size_t number_bins) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto histogram = utility::FixedSizeHistogram<arc_id_type>{ minimum, maximum, number_bins };

        for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
            const auto out_degree = graph.get_number_out_arcs(my_rank, local_node);
            histogram.add_data_point(out_degree);
        }

        return mpiPP::MPIAdvancedReductions::reduce_histogram(utility::Histogram<arc_id_type>{ std::move(histogram) });
    }
};
