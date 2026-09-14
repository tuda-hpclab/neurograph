/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Reciprocity.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "metrics/counting/NodeCounter.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <utility>
#include <vector>

std::pair<double, double> Reciprocity::compute_reciprocity(const DistributedGraph& graph) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto number_local_nodes = graph.get_number_local_nodes();

    auto local_number_arcs = distance_type{ 0 };
    auto local_number_mutual_arcs = distance_type{ 0 };

    // The in neighbors of the current node; kept outside the loop so that its storage is reused
    auto in_neighbors = std::vector<std::pair<mpi_rank_type, node_id_type>>{};

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto& in_arcs = graph.get_in_arcs(my_rank, node_id);

        in_neighbors.clear();
        in_neighbors.reserve(in_arcs.size());

        for (const auto& in_arc : in_arcs) {
            in_neighbors.emplace_back(in_arc.source_rank, in_arc.source_id);
        }

        // The arcs are stored in the unspecified order of the loaded hash map, so they must be sorted
        // before the opposite of an out arc can be looked up by binary search
        std::ranges::sort(in_neighbors);

        for (const auto& [target_rank, target_id, _] : graph.get_out_arcs(my_rank, node_id)) {
            if (target_rank == my_rank && target_id == node_id) {
                // A self arc is its own opposite, so counting it would only inflate the reciprocity
                continue;
            }

            ++local_number_arcs;

            // The arc (my_rank, node_id) -> (target_rank, target_id) is mutual iff the node has an
            // in arc from its target, which is stored locally next to the out arc
            if (std::ranges::binary_search(in_neighbors, std::pair{ target_rank, target_id })) {
                ++local_number_mutual_arcs;
            }
        }
    }

    const auto number_arcs = mpiPP::MPIReductions::reduce_sum(local_number_arcs);
    const auto number_mutual_arcs = mpiPP::MPIReductions::reduce_sum(local_number_mutual_arcs);
    const auto number_nodes = NodeCounter::count_nodes(graph);

    if (number_arcs == 0) {
        // Nothing can be reciprocal without arcs; this also covers every rank but the root one,
        // where the reductions above return zero
        return { 0.0, 0.0 };
    }

    const auto reciprocity = static_cast<double>(number_mutual_arcs) / static_cast<double>(number_arcs);

    // There is at least one arc between two distinct nodes, so there are at least two nodes
    const auto number_node_pairs = static_cast<double>(number_nodes) * static_cast<double>(number_nodes - 1);
    const auto density = static_cast<double>(number_arcs) / number_node_pairs;

    if (density >= 1.0) {
        // The graph is complete, where every arc is mutual by construction and the coefficient is undefined
        return { reciprocity, 0.0 };
    }

    const auto coefficient = (reciprocity - density) / (1.0 - density);

    return { reciprocity, coefficient };
}
