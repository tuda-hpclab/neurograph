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
#include "metrics/NodeCounter.h"
#include "utility/Vec3.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"

#include <vector>

/**
 * @brief Functions to compute the average, minimum, and maximum distance between all nodes.
 *		Functions must be called on all ranks simultaneously.
 */
class AllPairsDistances {
public:
    /**
     * @brief Computes the average pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The average distance
     */
    [[nodiscard]] static double compute_average_pair_distance(const DistributedGraph& graph) {
        const auto accumulated_distance = compute_local_sum_pair_distances(graph);
        const auto total_distance = mpiPP::MPIReductions::reduce_sum(accumulated_distance);

        const auto number_total_nodes = NodeCounter::count_nodes(graph);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return 0.0;
        }

        const auto number_total_pairs = (number_total_nodes * (number_total_nodes - 1)) / 2;
        const auto average_distance = total_distance / static_cast<double>(number_total_pairs);

        return average_distance;
    }

    /**
     * @brief Computes the maximum pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The maximum distance
     */
    [[nodiscard]] static double compute_maximum_pair_distance(const DistributedGraph& graph) {
        const auto maximum_distance = compute_local_maximum_pair_distances(graph);
        const auto global_maximum_distance = mpiPP::MPIReductions::reduce_max(maximum_distance);
        return global_maximum_distance;
    }

    /**
     * @brief Computes the minimum pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The minimum distance
     */
    [[nodiscard]] static double compute_minimum_pair_distance(const DistributedGraph& graph) {
        const auto minimum_distance = compute_local_minimum_pair_distances(graph);
        const auto global_minimum_distance = mpiPP::MPIReductions::reduce_min(minimum_distance);
        return global_minimum_distance;
    }

    /**
     * @brief Computes the average pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The average distance
     */
    [[nodiscard]] static double all_compute_average_pair_distance(const DistributedGraph& graph) {
        const auto accumulated_distance = compute_local_sum_pair_distances(graph);
        const auto total_distance = mpiPP::MPIReductions::all_reduce_sum(accumulated_distance);

        const auto number_total_nodes = NodeCounter::all_count_nodes(graph);

        const auto number_total_pairs = (number_total_nodes * (number_total_nodes - 1)) / 2;
        const auto average_distance = total_distance / static_cast<double>(number_total_pairs);

        return average_distance;
    }

    /**
     * @brief Computes the maximum pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The maximum distance
     */
    [[nodiscard]] static double all_compute_maximum_pair_distance(const DistributedGraph& graph) {
        const auto maximum_distance = compute_local_maximum_pair_distances(graph);
        const auto global_maximum_distance = mpiPP::MPIReductions::all_reduce_max(maximum_distance);
        return global_maximum_distance;
    }

    /**
     * @brief Computes the minimum pairwise distance between nodes in the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     *		Complexity is O(A*A/R + R), possibly O(A*A/R + log R).
     * @param graph The graph
     * @return The minimum distance
     */
    [[nodiscard]] static double all_compute_minimum_pair_distance(const DistributedGraph& graph) {
        const auto minimum_distance = compute_local_minimum_pair_distances(graph);
        const auto global_minimum_distance = mpiPP::MPIReductions::all_reduce_min(minimum_distance);
        return global_minimum_distance;
    }

private:
    [[nodiscard]] static double compute_local_sum_pair_distances(const DistributedGraph& graph) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);

        const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto ranks_to_compute = number_ranks / 2;

        const auto get_next_rank = [number_ranks, my_rank](const mpi_rank_type offset) {
            return utility::save_cast<mpi_rank_type>((utility::save_cast<std::size_t>(my_rank + offset)) % number_ranks);
        };

        auto local_sum = get_summed_distance(graph, my_rank, node_distribution[utility::save_cast<std::size_t>(my_rank)]);

        for (auto offset = 1U; offset <= ranks_to_compute; ++offset) {
            const auto other_rank = get_next_rank(utility::save_cast<mpi_rank_type>(offset));

            if (number_ranks % 2 == 0 && offset == ranks_to_compute && my_rank > other_rank) {
                continue;
            }

            local_sum += get_summed_distance(graph, other_rank, node_distribution[utility::save_cast<std::size_t>(other_rank)]);
        }

        return local_sum;
    }

    [[nodiscard]] static double compute_local_maximum_pair_distances(const DistributedGraph& graph) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);

        const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto ranks_to_compute = number_ranks / 2;

        const auto get_next_rank = [number_ranks, my_rank](const mpi_rank_type offset) {
            return utility::save_cast<mpi_rank_type>((utility::save_cast<std::size_t>(my_rank + offset)) % number_ranks);
        };

        auto local_maximum = get_maximum_distance(graph, my_rank, node_distribution[utility::save_cast<std::size_t>(my_rank)]);

        for (auto offset = 1U; offset <= ranks_to_compute; ++offset) {
            const auto other_rank = get_next_rank(utility::save_cast<mpi_rank_type>(offset));

            if (number_ranks % 2 == 0 && offset == ranks_to_compute && my_rank > other_rank) {
                continue;
            }

            local_maximum = std::max(local_maximum, get_maximum_distance(graph, other_rank, node_distribution[utility::save_cast<std::size_t>(other_rank)]));
        }

        return local_maximum;
    }

    [[nodiscard]] static double compute_local_minimum_pair_distances(const DistributedGraph& graph) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);

        const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto ranks_to_compute = number_ranks / 2;

        const auto get_next_rank = [number_ranks, my_rank](const mpi_rank_type offset) {
            return utility::save_cast<mpi_rank_type>((utility::save_cast<std::size_t>(my_rank + offset)) % number_ranks);
        };

        auto local_minimum = get_minimum_distance(graph, my_rank, node_distribution[utility::save_cast<std::size_t>(my_rank)]);

        for (auto offset = 1U; offset <= ranks_to_compute; ++offset) {
            const auto other_rank = get_next_rank(utility::save_cast<mpi_rank_type>(offset));

            if (number_ranks % 2 == 0 && offset == ranks_to_compute && my_rank > other_rank) {
                continue;
            }

            local_minimum = std::min(local_minimum, get_minimum_distance(graph, other_rank, node_distribution[utility::save_cast<std::size_t>(other_rank)]));
        }

        return local_minimum;
    }

    [[nodiscard]] static double get_summed_distance(const DistributedGraph& graph, const mpi_rank_type other_rank, const node_id_type number_foreign_nodes) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto summed_distance = 0.0;

        for (auto foreign_node = node_id_type(0); foreign_node < number_foreign_nodes; ++foreign_node) {
            const auto foreign_position = graph.get_node_position(other_rank, foreign_node);

            for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
                if (my_rank == other_rank && local_node == foreign_node) {
                    break;
                }

                const auto own_position = graph.get_node_position(my_rank, local_node);
                const auto distance = (foreign_position - own_position).calculate_2_norm();
                summed_distance += distance;
            }
        }

        return summed_distance;
    }

    [[nodiscard]] static double get_maximum_distance(const DistributedGraph& graph, const mpi_rank_type other_rank, const node_id_type number_foreign_nodes) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto maximum_distance = 0.0;

        for (auto foreign_node = node_id_type(0); foreign_node < number_foreign_nodes; ++foreign_node) {
            const auto foreign_position = graph.get_node_position(other_rank, foreign_node);

            for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
                if (my_rank == other_rank && local_node == foreign_node) {
                    break;
                }

                const auto own_position = graph.get_node_position(my_rank, local_node);
                const auto distance = (foreign_position - own_position).calculate_2_norm();
                maximum_distance = std::max(maximum_distance, distance);
            }
        }

        return maximum_distance;
    }

    [[nodiscard]] static double get_minimum_distance(const DistributedGraph& graph, const mpi_rank_type other_rank, const node_id_type number_foreign_nodes) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto minimum_distance = std::numeric_limits<double>::max();

        for (auto foreign_node = node_id_type(0); foreign_node < number_foreign_nodes; ++foreign_node) {
            const auto foreign_position = graph.get_node_position(other_rank, foreign_node);

            for (auto local_node = node_id_type(0); local_node < number_local_nodes; ++local_node) {
                if (my_rank == other_rank && local_node == foreign_node) {
                    break;
                }

                const auto own_position = graph.get_node_position(my_rank, local_node);
                const auto distance = (foreign_position - own_position).calculate_2_norm();
                minimum_distance = std::min(minimum_distance, distance);
            }
        }

        return minimum_distance;
    }
};
