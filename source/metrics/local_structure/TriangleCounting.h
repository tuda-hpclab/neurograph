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

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>

/**
 * @brief Counts the triangles through a single node, i.e., the closed triangles and the ones that
 *		the neighborhood of the node allows. The two metrics that build on those counts differ only
 *		in how they combine them: the clustering coefficient (Clustering) averages the ratios of the
 *		nodes, the transitivity (Transitivity) divides the sums.
 */
class TriangleCounting {
public:
    /**
     * @brief Counts the triangles of the form i->j->k->i for three pairwise distinct nodes, where i is
     *		the given node. Arcs from the node to itself are ignored, and a mutual arc i<->j is not
     *		counted in the denominator, because it can never be part of such a triangle.
     *		Must be called on the MPI rank that owns the node; the neighborhoods of the other two nodes
     *		are fetched from their ranks if needed.
     *		Complexity is O(d_out * d * d_in) in the worst case, where d is the maximum degree.
     * @param graph The graph
     * @param node_id The local id of the node
     * @return The number of closed triangles and the number of triangles the neighborhood allows,
     *		both zero if the node has no such triangle
     */
    [[nodiscard]] static std::pair<std::uint64_t, std::uint64_t> count_triangles_node(const DistributedGraph& graph, const node_id_type node_id) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto number_closed_triangles = std::uint64_t{ 0 };

        const auto& in_arcs = graph.get_in_arcs(my_rank, node_id);
        const auto number_in_neighbors = contains_other(in_arcs, my_rank, node_id) ? in_arcs.size() - 1 : in_arcs.size();

        const auto& out_arcs = graph.get_out_arcs(my_rank, node_id);
        const auto number_out_neighbors = contains_other(out_arcs, my_rank, node_id) ? out_arcs.size() - 1 : out_arcs.size();

        auto number_all_triangles = utility::safe_cast<std::uint64_t>(number_in_neighbors * number_out_neighbors);
        if (number_all_triangles == 0) {
            return { 0, 0 };
        }

        for (const auto& [target_rank, target_id, weight] : out_arcs) {
            if (my_rank == target_rank && target_id == node_id) {
                // Ignore arc from node_id to itself
                continue;
            }

            const auto other_out_arcs = graph.get_out_arcs(target_rank, target_id);

            if (contains_other(other_out_arcs, my_rank, node_id)) {
                // There is an arc from (my_rank, node_id) to (target_rank, target_id) and back --> we don't want to have those in the
                // denominator
                --number_all_triangles;
            }

            const auto number_matches = count_matches(in_arcs, other_out_arcs, my_rank, node_id, target_rank, target_id);
            number_closed_triangles += number_matches;
        }

        if (number_all_triangles == std::uint64_t{ 0 }) {
            return { 0, 0 };
        }

        return { number_closed_triangles, number_all_triangles };
    }

    /**
     * @brief Counts the triangles of the form i->j->k<-i for three pairwise distinct nodes, where i is
     *		the given node, i.e., the variant that only looks at the out neighborhoods.
     *		Arcs from the node to itself are ignored.
     *		Must be called on the MPI rank that owns the node; the neighborhoods of the other two nodes
     *		are fetched from their ranks if needed.
     *		Complexity is O(d_out * d * d_out) in the worst case, where d is the maximum degree.
     * @param graph The graph
     * @param node_id The local id of the node
     * @return The number of closed triangles and the number of triangles the neighborhood allows,
     *		both zero if the node has no such triangle
     */
    [[nodiscard]] static std::pair<std::uint64_t, std::uint64_t> count_triangles_node_2(const DistributedGraph& graph, const node_id_type node_id) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

        auto number_closed_triangles = std::uint64_t{ 0 };

        const auto& out_arcs = graph.get_out_arcs(my_rank, node_id);
        const auto number_neighbors = contains_other(out_arcs, my_rank, node_id) ? out_arcs.size() - 1 : out_arcs.size();

        const auto number_all_triangles = utility::safe_cast<std::uint64_t>(number_neighbors * (number_neighbors - 1));
        if (number_all_triangles == 0) {
            return { 0, 0 };
        }

        for (const auto& [target_rank, target_id, weight] : out_arcs) {
            if (my_rank == target_rank && target_id == node_id) {
                // Ignore arc from node_id to itself
                continue;
            }

            const auto other_out_arcs = graph.get_out_arcs(target_rank, target_id);

            const auto number_matches = count_matches(out_arcs, other_out_arcs, my_rank, node_id, target_rank, target_id);
            number_closed_triangles += number_matches;
        }

        return { number_closed_triangles, number_all_triangles };
    }

private:
    template <typename ArcType>
    [[nodiscard]] static bool contains_other(const std::span<const ArcType> arcs, const mpi_rank_type my_rank, const node_id_type node_id) {
        const auto is_arc_of_my_rank_and_node_id = [node_id, my_rank](const auto& arc) {
            const auto& [other_rank, other_id, _] = arc;
            return other_rank == my_rank && other_id == node_id;
        };
        return std::ranges::any_of(arcs, is_arc_of_my_rank_and_node_id);
    }

    template <typename ArcType>
    [[nodiscard]] static std::uint64_t count_matches(const std::span<const ArcType> other_arcs, const std::span<const OutArc> out_arcs,
                                                     const mpi_rank_type my_mpi_rank, const node_id_type my_node_id, const mpi_rank_type out_arc_mpi_rank,
                                                     const node_id_type out_arc_node_id) noexcept {
        auto counter = std::uint64_t{ 0 };

        for (const auto& [rank_1, id_1, _1] : out_arcs) {

            const auto other_bad_id = id_1 == out_arc_node_id;

            if (const auto other_bad_rank = rank_1 == out_arc_mpi_rank; other_bad_rank && other_bad_id) {
                // That's an in arc from the node to itself; we ignore those cases
                continue;
            }

            const auto my_bad_id = id_1 == my_node_id;

            if (const auto my_bad_rank = rank_1 == my_mpi_rank; my_bad_rank && my_bad_id) {
                // That's an in arc from the node to itself; we ignore those cases
                continue;
            }

            for (const auto& [rank_2, id_2, _2] : other_arcs) {
                const auto rank_match = rank_1 == rank_2;
                const auto id_match = id_1 == id_2;

                if (rank_match && id_match) {
                    ++counter;
                    break;
                }
            }
        }

        return counter;
    }
};
