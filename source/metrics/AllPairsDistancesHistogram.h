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

#include "cpp-utility/Cast.hpp"

#include "graph/DistributedGraph.h"
#include "metrics/NodeCounter.h"
#include "utility/Histogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <vector>

/**
 * @brief Functions to compute the distances between any two nodes and to count them in
 *		a distances histogram. Functions must be called on all ranks simultaneously.
 */
class AllPairsDistancesHistogram {
public:
    /**
     * @brief Computes a histogram of the pairwise distances between nodes in the graph with a fixed bin width.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @param bin_width The width of the bins in the histogram
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<double> compute_pair_distances_fixed_bin_width(const DistributedGraph& graph, const double bin_width) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);

        const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto ranks_to_compute = number_ranks / 2;

        const auto get_next_rank = [number_ranks, my_rank](const mpi_rank_type offset) {
            return utility::save_cast<mpi_rank_type>(utility::save_cast<std::size_t>(my_rank + offset) % number_ranks);
        };

        auto histogram = FixedWidthHistogram<double>(bin_width);

        add_to_histogram(graph, histogram, my_rank, node_distribution[utility::save_cast<std::size_t>(my_rank)]);

        for (auto offset = 1U; offset <= ranks_to_compute; ++offset) {
            const auto other_rank = get_next_rank(utility::save_cast<mpi_rank_type>(offset));

            if (number_ranks % 2 == 0 && offset == ranks_to_compute && my_rank > other_rank) {
                continue;
            }

            add_to_histogram(graph, histogram, other_rank, node_distribution[utility::save_cast<std::size_t>(other_rank)]);
        }

        auto result = Histogram<double>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

    /**
     * @brief Computes a histogram of the pairwise distances between nodes in the graph with a fixed bin count.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @param minimum The minimum length
     * @param maximum The maximum length
     * @param number_bins The number of bins in the histogram
     * @exception Throws an exception if any arc is longer than the maximum or shorter than the minimum
     * @return The histogram on MPI rank 0
     */
    [[nodiscard]] static Histogram<double> compute_pair_distances_fixed_bin_count(const DistributedGraph& graph, const double minimum, const double maximum,
                                                                                  const std::size_t number_bins) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);

        const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto ranks_to_compute = number_ranks / 2;

        const auto get_next_rank = [number_ranks, my_rank](const mpi_rank_type offset) {
            return utility::save_cast<mpi_rank_type>((utility::save_cast<std::size_t>(my_rank + offset)) % number_ranks);
        };

        auto histogram = FixedSizeHistogram<double>{ minimum, maximum, number_bins };

        add_to_histogram(graph, histogram, my_rank, node_distribution[utility::save_cast<std::size_t>(my_rank)]);

        for (auto offset = 1U; offset <= ranks_to_compute; ++offset) {
            const auto other_rank = get_next_rank(utility::save_cast<mpi_rank_type>(offset));

            if (number_ranks % 2 == 0 && offset == ranks_to_compute && my_rank > other_rank) {
                continue;
            }

            add_to_histogram(graph, histogram, other_rank, node_distribution[utility::save_cast<std::size_t>(other_rank)]);
        }

        auto result = Histogram<double>{ std::move(histogram) };
        result.sum_on_rank_0();

        return result;
    }

private:
    template <typename HistogramType>
    static void add_to_histogram(const DistributedGraph& graph, HistogramType& histogram, const mpi_rank_type other_rank,
                                 const node_id_type number_foreign_nodes) {

        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        const auto number_local_nodes = graph.get_number_local_nodes();

        for (auto foreign_node = node_id_type(0); foreign_node < number_foreign_nodes; ++foreign_node) {
            const auto foreign_position = graph.get_node_position(other_rank, foreign_node);

            for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
                if (my_rank == other_rank && local_node == foreign_node) {
                    break;
                }

                const auto own_position = graph.get_node_position(my_rank, local_node);
                const auto distance = (foreign_position - own_position).calculate_2_norm();
                histogram.add_data_point(distance);
            }
        }
    }
};
