/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RichClub.h"

#include "Types.h"

#include "graph/DistributedGraph.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

/**
 * The degree of a node. Wider than arc_id_type, because it is the sum of two counts of that type.
 */
using degree_type = std::uint64_t;

/**
 * @brief Returns the degree of the specified node, i.e., its number of in arcs plus its number of out
 *		arcs. Might perform communication via MPI for a node of another rank.
 * @param graph The graph
 * @param mpi_rank The MPI rank (can be the current MPI rank)
 * @param node_id The node id on the MPI rank
 * @return The node's degree
 */
[[nodiscard]] degree_type compute_degree(const DistributedGraph& graph, const mpi_rank_type mpi_rank, const node_id_type node_id) {
    const auto number_in_arcs = static_cast<degree_type>(graph.get_number_in_arcs(mpi_rank, node_id));
    const auto number_out_arcs = static_cast<degree_type>(graph.get_number_out_arcs(mpi_rank, node_id));

    return number_in_arcs + number_out_arcs;
}

} // namespace

RichClubResult RichClub::compute_rich_club_coefficients(const DistributedGraph& graph) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto number_local_nodes = graph.get_number_local_nodes();

    auto local_degrees = std::vector<degree_type>(number_local_nodes, degree_type{ 0 });
    auto local_maximum_degree = degree_type{ 0 };

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto degree = compute_degree(graph, my_rank, node_id);

        local_degrees[node_id] = degree;
        local_maximum_degree = std::max(local_maximum_degree, degree);
    }

    // All ranks contribute a histogram over the degrees to the reductions below, so all of them need
    // the same length and therefore the largest degree of the whole graph
    const auto maximum_degree = mpiPP::MPIReductions::all_reduce_max(local_maximum_degree);

    if (maximum_degree == 0) {
        // No node has an arc, so no club has two members that could be connected. All ranks agree on
        // that, so returning here leaves no collective unmatched
        return {};
    }

    const auto number_degrees = utility::safe_cast<std::size_t>(maximum_degree) + 1;

    // <local_number_nodes_by_degree>[d] == k indicates that k local nodes have a degree of exactly d
    auto local_number_nodes_by_degree = std::vector<global_node_id_type>(number_degrees, global_node_id_type{ 0 });

    // <local_number_arcs_by_minimum_degree>[d] == k indicates that k local arcs run between two
    // distinct nodes the smaller degree of which is exactly d. Such an arc runs inside the club of the
    // threshold t iff both of its endpoints have a degree larger than t, i.e., iff t is smaller than d
    auto local_number_arcs_by_minimum_degree = std::vector<distance_type>(number_degrees, distance_type{ 0 });

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto degree = local_degrees[node_id];
        ++local_number_nodes_by_degree[utility::safe_cast<std::size_t>(degree)];

        // Every arc is stored as an out arc of exactly one node, so iterating the out arcs of all
        // local nodes on every rank visits every arc of the graph exactly once
        for (const auto& [target_rank, target_id, _] : graph.get_out_arcs(my_rank, node_id)) {
            if (target_rank == my_rank && target_id == node_id) {
                // An arc from a node to itself runs inside one member of the club instead of between
                // two of them, and the pair it would belong to is not part of the denominator either
                continue;
            }

            const auto target_degree = (target_rank == my_rank) ? local_degrees[target_id] : compute_degree(graph, target_rank, target_id);
            const auto minimum_degree = std::min(degree, target_degree);

            ++local_number_arcs_by_minimum_degree[utility::safe_cast<std::size_t>(minimum_degree)];
        }
    }

    const auto number_nodes_by_degree = mpiPP::MPIReductions::reduce_componentwise_sum(local_number_nodes_by_degree);
    const auto number_arcs_by_minimum_degree = mpiPP::MPIReductions::reduce_componentwise_sum(local_number_arcs_by_minimum_degree);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return {};
    }

    // The club of a threshold consists of the nodes of a larger degree, so its size and its number of
    // arcs are the suffix sums of the two histograms, i.e., <club_sizes>[t] sums up all degrees above t
    auto club_sizes = std::vector<global_node_id_type>(number_degrees, global_node_id_type{ 0 });
    auto club_arc_counts = std::vector<distance_type>(number_degrees, distance_type{ 0 });

    for (auto degree = number_degrees - 1; degree > 0; --degree) {
        club_sizes[degree - 1] = club_sizes[degree] + number_nodes_by_degree[degree];
        club_arc_counts[degree - 1] = club_arc_counts[degree] + number_arcs_by_minimum_degree[degree];
    }

    auto result = RichClubResult{};

    // A club of a single node has no pair of distinct nodes and therefore no density. The clubs only
    // shrink as the threshold grows, so this cuts off the thresholds from the largest useful one on;
    // the club of the largest degree is empty, which ends the loop at the latest
    for (auto threshold = std::size_t{ 0 }; threshold < number_degrees && club_sizes[threshold] >= 2; ++threshold) {
        const auto club_size = static_cast<double>(club_sizes[threshold]);
        const auto number_pairs = club_size * (club_size - 1.0);

        result.coefficients.push_back(static_cast<double>(club_arc_counts[threshold]) / number_pairs);
        result.club_sizes.push_back(club_sizes[threshold]);
        result.club_arc_counts.push_back(club_arc_counts[threshold]);
    }

    return result;
}
