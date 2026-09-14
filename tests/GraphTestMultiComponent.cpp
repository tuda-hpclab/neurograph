/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GraphTest.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <string>
#include <utility>
#include <vector>

namespace {
/** The number of nodes each rank holds in the multi-component graph. */
constexpr auto number_multi_component_nodes_per_rank = node_id_type{ 3 };

/** One arc of the multi-component graph, see GraphTest::get_multi_component_four_rank_graph. */
struct MultiComponentArc {
    mpi_rank_type source_rank{};
    node_id_type source_node_id{};

    mpi_rank_type target_rank{};
    node_id_type target_node_id{};

    weight_type weight{};
};

/**
 * @brief Returns the arcs of the multi-component graph. The components are the ring through the first
 *		node of every rank, the pair (1, 1) <-> (1, 2), the pair (2, 1) <-> (3, 1), and the four
 *		remaining single nodes; all other arcs lie on no cycle at all.
 * @return The arcs
 */
[[nodiscard]] std::vector<MultiComponentArc> get_multi_component_arcs() {
    return std::vector<MultiComponentArc>{
        // The only component that spans all ranks
        { .source_rank = 0, .source_node_id = 0, .target_rank = 1, .target_node_id = 0, .weight = 1 },
        { .source_rank = 1, .source_node_id = 0, .target_rank = 2, .target_node_id = 0, .weight = 2 },
        { .source_rank = 2, .source_node_id = 0, .target_rank = 3, .target_node_id = 0, .weight = 3 },
        { .source_rank = 3, .source_node_id = 0, .target_rank = 0, .target_node_id = 0, .weight = 4 },

        // A component within one rank
        { .source_rank = 1, .source_node_id = 1, .target_rank = 1, .target_node_id = 2, .weight = 5 },
        { .source_rank = 1, .source_node_id = 2, .target_rank = 1, .target_node_id = 1, .weight = 6 },

        // A component across two ranks
        { .source_rank = 2, .source_node_id = 1, .target_rank = 3, .target_node_id = 1, .weight = 7 },
        { .source_rank = 3, .source_node_id = 1, .target_rank = 2, .target_node_id = 1, .weight = 8 },

        // A chain into the component within rank 1, whose nodes are trimmed away one after the other
        { .source_rank = 0, .source_node_id = 1, .target_rank = 0, .target_node_id = 2, .weight = 9 },
        { .source_rank = 0, .source_node_id = 2, .target_rank = 1, .target_node_id = 1, .weight = 10 },

        // A path out of that component into the ring
        { .source_rank = 1, .source_node_id = 2, .target_rank = 2, .target_node_id = 2, .weight = 11 },
        { .source_rank = 2, .source_node_id = 2, .target_rank = 2, .target_node_id = 0, .weight = 12 },

        // A path out of the ring into the component across ranks 2 and 3
        { .source_rank = 3, .source_node_id = 0, .target_rank = 3, .target_node_id = 2, .weight = 13 },
        { .source_rank = 3, .source_node_id = 2, .target_rank = 3, .target_node_id = 1, .weight = 14 },
    };
}

/**
 * @brief Returns the nodes of the multi-component graph. Their positions, areas, and signal types play
 *		no role for the connectivity, so all nodes share one area and one signal type.
 * @param number_nodes The number of nodes to create
 * @param first_global_id The global id of the first node, which only keeps the positions apart
 * @return The nodes
 */
[[nodiscard]] LoadedNodes get_multi_component_nodes(const node_id_type number_nodes, const node_id_type first_global_id) {
    auto positions = std::vector<Vec3d>{};
    positions.reserve(number_nodes);

    for (auto node_id = node_id_type{ 0 }; node_id < number_nodes; ++node_id) {
        const auto global_id = static_cast<double>(first_global_id + node_id);
        positions.emplace_back(global_id, 0.5 * global_id, 1.0);
    }

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");

    auto area_names_indices = std::vector<node_id_type>(number_nodes, node_id_type{ 0 });
    auto signal_types_indices = std::vector<node_id_type>(number_nodes, node_id_type{ 0 });

    return LoadedNodes{ std::move(positions), std::move(area_names), std::move(signal_types), std::move(area_names_indices), std::move(signal_types_indices) };
}
} // namespace

DistributedGraph GraphTest::get_multi_component_four_rank_graph() {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = number_multi_component_nodes_per_rank;

    auto loaded_nodes = get_multi_component_nodes(number_local_nodes, utility::safe_cast<node_id_type>(my_rank) * number_local_nodes);

    auto in_arcs = LoadedArcs(number_local_nodes);
    auto out_arcs = LoadedArcs(number_local_nodes);

    for (const auto& [source_rank, source_node_id, target_rank, target_node_id, weight] : get_multi_component_arcs()) {
        if (source_rank == my_rank) {
            out_arcs[source_node_id][{ target_rank, target_node_id }] = weight;
        }

        // Unlike generate_arcs above, the in arcs mirror the out arcs, so both directions describe the
        // same graph even though none of the arcs is bidirectional
        if (target_rank == my_rank) {
            in_arcs[target_node_id][{ source_rank, source_node_id }] = weight;
        }
    }

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_multi_component_four_rank_graph_on_one_rank() {
    const auto number_nodes = 4 * number_multi_component_nodes_per_rank;

    auto loaded_nodes = get_multi_component_nodes(number_nodes, node_id_type{ 0 });

    auto in_arcs = LoadedArcs(number_nodes);
    auto out_arcs = LoadedArcs(number_nodes);

    const auto on_one_rank = [](const mpi_rank_type mpi_rank, const node_id_type node_id) {
        return utility::safe_cast<node_id_type>(mpi_rank) * number_multi_component_nodes_per_rank + node_id;
    };

    for (const auto& [source_rank, source_node_id, target_rank, target_node_id, weight] : get_multi_component_arcs()) {
        const auto source = on_one_rank(source_rank, source_node_id);
        const auto target = on_one_rank(target_rank, target_node_id);

        out_arcs[source][{ 0, target }] = weight;
        in_arcs[target][{ 0, source }] = weight;
    }

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}
