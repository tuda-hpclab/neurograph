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

#include "graph/ArcTransformer.h"
#include "graph/DistributedGraph.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr auto number_standard_nodes = node_id_type{ 10 };

struct RankAssignment {
    mpi_rank_type canonical_rank;
    mpi_rank_type predecessor;
    mpi_rank_type successor;
};

[[nodiscard]] RankAssignment get_rank_assignment(const mpi_rank_type rank) {
    // The canonical rank selects the corresponding local data from the four-rank graph. The
    // predecessor and the successor only matter for the standard graph: there the ranks 0, 1, 4,
    // and 6 form a ring and the ranks 2 and 5 form a second, two-rank one.
    switch (rank) {
    case 0:
        return { 0, 6, 1 };
    case 1:
        return { 1, 0, 4 };
    case 2:
        return { 0, 5, 5 };
    case 4:
        return { 2, 1, 6 };
    case 5:
        return { 1, 2, 2 };
    case 6:
        return { 3, 4, 0 };
    default:
        // Rank 3 uses the fourth canonical node set, but receives the source/sink arcs below.
        return { 3, 3, 3 };
    }
}

[[nodiscard]] Vec3d get_position_offset(const mpi_rank_type rank) {
    // Ranks 2, 5, and 6 repeat the canonical node sets of ranks 0, 1, and 3. Translating their
    // copies keeps every node of the graph at a position of its own.
    switch (rank) {
    case 2:
        return { 12.0, 0.0, 0.0 };
    case 5:
        return { 0.0, 12.0, 0.0 };
    case 6:
        return { 0.0, 0.0, 12.0 };
    default:
        return { 0.0, 0.0, 0.0 };
    }
}

[[nodiscard]] LoadedNodes get_standard_nodes(const mpi_rank_type canonical_rank) {
    auto positions = std::vector<Vec3d>{};
    auto area_names = std::vector<std::string>{};
    auto signal_types = std::vector<std::string>{ "excitatory", "inhibitory" };
    auto area_names_indices = std::vector<node_id_type>{};
    auto signal_types_indices = std::vector<node_id_type>{};

    switch (canonical_rank) {
    case 0:
        positions = { { 0.1, 9.8, 4.7 }, { 9.6, 5.7, 1.4 }, { 0.3, 3.9, 8.0 }, { 7.9, 2.8, 9.6 }, { 2.2, 4.9, 0.0 }, { 0.9, 5.2, 8.6 }, { 0.6, 5.8, 5.2 }, { 8.2, 2.0, 8.8 }, { 8.2, 4.8, 6.6 }, { 7.4, 4.4, 3.4 } };
        area_names = { "area_1", "area_2", "area_3" };
        area_names_indices = { 0, 1, 1, 0, 1, 1, 0, 1, 0, 2 };
        signal_types_indices = { 0, 0, 1, 0, 0, 0, 1, 0, 1, 0 };
        break;
    case 1:
        positions = { { 0.40, 0.31, 0.64 }, { 0.90, 0.83, 0.47 }, { 0.78, 0.45, 0.45 }, { 0.41, 0.22, 0.02 }, { 0.19, 0.93, 0.04 }, { 0.21, 0.28, 0.84 }, { 0.00, 0.41, 0.88 }, { 0.12, 0.55, 0.02 }, { 0.22, 0.91, 0.19 }, { 0.43, 0.99, 0.61 } };
        area_names = { "area_1", "area_2", "area_4" };
        signal_types = { "inhibitory", "excitatory" };
        area_names_indices = { 0, 0, 1, 0, 0, 2, 2, 2, 2, 0 };
        signal_types_indices = { 1, 1, 1, 0, 1, 0, 1, 0, 1, 1 };
        break;
    case 2:
        positions = { { 0.18, 0.26, 1.00 }, { 0.72, 0.59, 0.06 }, { 0.46, 0.67, 0.75 }, { 0.56, 0.44, 0.46 }, { 0.48, 0.12, 0.64 }, { 0.17, 0.79, 0.01 }, { 0.65, 0.17, 1.00 }, { 0.58, 0.87, 0.38 }, { 0.05, 0.57, 0.33 }, { 0.96, 0.44, 0.95 } };
        area_names = { "area_1", "area_4", "area_5" };
        area_names_indices = { 0, 1, 0, 0, 1, 2, 0, 2, 2, 2 };
        signal_types_indices = { 0, 1, 1, 0, 1, 0, 1, 0, 1, 1 };
        break;
    default:
        positions = { { 0.22, 0.29, 0.89 }, { 0.99, 0.51, 0.16 }, { 0.99, 0.72, 0.75 }, { 0.71, 0.01, 0.71 }, { 0.93, 0.68, 0.42 }, { 0.98, 0.47, 0.17 }, { 0.73, 0.48, 0.93 }, { 0.91, 0.35, 0.43 }, { 0.28, 0.12, 0.09 }, { 0.08, 0.67, 0.17 } };
        area_names = { "area_2", "area_4" };
        area_names_indices = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 0 };
        signal_types_indices = { 0, 1, 1, 1, 1, 1, 0, 0, 0, 1 };
        break;
    }

    return LoadedNodes{ std::move(positions), std::move(area_names), std::move(signal_types), std::move(area_names_indices),
                        std::move(signal_types_indices) };
}

[[nodiscard]] LoadedNodes get_full_nodes(const mpi_rank_type canonical_rank) {
    auto nodes = get_standard_nodes(canonical_rank);

    // These are the same non-uniform local node sets as in get_full_four_rank_graph().
    switch (canonical_rank) {
    case 0:
        nodes.positions.resize(8);
        nodes.area_names_ind = { 0, 1, 1, 0, 1, 1, 0, 2 };
        nodes.signal_types_ind = { 0, 0, 1, 0, 1, 0, 1, 0 };
        break;
    case 1:
        nodes.positions.resize(9);
        nodes.area_names_ind.resize(9);
        nodes.signal_types_ind = { 1, 1, 1, 0, 1, 1, 0, 1, 1 };
        break;
    case 2:
        nodes.positions.resize(9);
        nodes.area_names_ind = { 0, 1, 0, 0, 2, 0, 2, 2, 2 };
        nodes.signal_types_ind = { 0, 1, 1, 0, 0, 1, 0, 1, 1 };
        break;
    default:
        break;
    }

    return nodes;
}

[[nodiscard]] LoadedNodes get_nodes_of_rank(const mpi_rank_type rank, const bool full) {
    const auto canonical_rank = get_rank_assignment(rank).canonical_rank;
    auto nodes = full ? get_full_nodes(canonical_rank) : get_standard_nodes(canonical_rank);

    const auto offset = get_position_offset(rank);
    for (auto& position : nodes.positions) {
        position += offset;
    }

    return nodes;
}

[[nodiscard]] std::pair<LoadedArcs, LoadedArcs> get_standard_arcs(const mpi_rank_type owner_rank, const mpi_rank_type predecessor,
                                                                  const mpi_rank_type successor) {
    using ArcCollection = LoadedArcs::value_type;

    const auto arc = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{
        ArcCollection{ arc(owner_rank, 1, 1) },
        ArcCollection{ arc(owner_rank, 3, 1), arc(owner_rank, 9, 2), arc(owner_rank, 4, 1) },
        ArcCollection{ arc(owner_rank, 7, 1), arc(owner_rank, 4, 2) },
        ArcCollection{ arc(owner_rank, 2, 1), arc(owner_rank, 7, 2), arc(owner_rank, 8, 1) },
        ArcCollection{ arc(owner_rank, 9, 1), arc(owner_rank, 2, 2) },
        ArcCollection{ arc(owner_rank, 5, 1), arc(owner_rank, 3, 2) },
        ArcCollection{ arc(owner_rank, 0, 1), arc(owner_rank, 4, 2), arc(owner_rank, 1, 3), arc(owner_rank, 5, 9), arc(owner_rank, 6, 2) },
        ArcCollection{ arc(owner_rank, 3, 1), arc(owner_rank, 5, 2) },
        ArcCollection{ arc(owner_rank, 4, 1), arc(owner_rank, 0, 2) },
        ArcCollection{ arc(owner_rank, 6, 1), arc(owner_rank, 8, 2), arc(owner_rank, 0, 1) },
    };

    auto out_arcs = LoadedArcs{
        ArcCollection{ arc(owner_rank, 6, 1), arc(owner_rank, 8, 2), arc(owner_rank, 9, 1) },
        ArcCollection{ arc(owner_rank, 0, 1), arc(owner_rank, 6, 3) },
        ArcCollection{ arc(owner_rank, 3, 1), arc(owner_rank, 4, 2) },
        ArcCollection{ arc(owner_rank, 1, 1), arc(owner_rank, 5, 2), arc(owner_rank, 7, 1) },
        ArcCollection{ arc(owner_rank, 1, 1), arc(owner_rank, 2, 2), arc(owner_rank, 6, 2), arc(owner_rank, 8, 1) },
        ArcCollection{ arc(owner_rank, 5, 1), arc(owner_rank, 6, 9), arc(owner_rank, 7, 2) },
        ArcCollection{ arc(owner_rank, 6, 2), arc(owner_rank, 9, 1) },
        ArcCollection{ arc(owner_rank, 2, 1), arc(owner_rank, 3, 2) },
        ArcCollection{ arc(owner_rank, 3, 1), arc(owner_rank, 9, 2) },
        ArcCollection{ arc(owner_rank, 1, 2), arc(owner_rank, 4, 1) },
    };

    for (auto node_id = node_id_type{ 0 }; node_id < number_standard_nodes; ++node_id) {
        in_arcs[node_id][{ predecessor, node_id }] = 11;
        out_arcs[node_id][{ successor, node_id }] = 11;
    }

    return { std::move(in_arcs), std::move(out_arcs) };
}

[[nodiscard]] std::pair<LoadedArcs, LoadedArcs> get_source_sink_arcs(const node_id_type number_nodes) {
    auto in_arcs = LoadedArcs(number_nodes);
    auto out_arcs = LoadedArcs(number_nodes);

    // Nodes 0, 1, and 2 are isolated; 3, 4, and 5 are pure sinks; 6 through 9 are pure sources.
    const auto add_arc = [&in_arcs, &out_arcs](const node_id_type source, const node_id_type target, const weight_type weight) {
        out_arcs[source][{ 3, target }] = weight;
        in_arcs[target][{ 3, source }] = weight;
    };
    add_arc(6, 3, 1);
    add_arc(7, 4, 2);
    add_arc(8, 5, 3);
    add_arc(9, 3, 4);

    return { std::move(in_arcs), std::move(out_arcs) };
}

[[nodiscard]] DistributedGraph make_standard_seven_rank_graph(const bool undirected_unit_weight) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto assignment = get_rank_assignment(my_rank);
    auto loaded_nodes = get_nodes_of_rank(my_rank, false);

    auto arcs = my_rank == 3 ? get_source_sink_arcs(number_standard_nodes)
                             : get_standard_arcs(my_rank, assignment.predecessor, assignment.successor);
    auto& [in_arcs, out_arcs] = arcs;

    if (!undirected_unit_weight) {
        return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
    }

    auto synchronized_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));
    auto unit_arcs = ArcTransformer::all_weights_one(std::move(synchronized_arcs));
    return DistributedGraph::construct_graph(std::move(loaded_nodes), unit_arcs, unit_arcs);
}

[[nodiscard]] node_id_type get_full_node_count(const mpi_rank_type rank) {
    switch (rank) {
    case 0:
    case 2:
        return 8;
    case 1:
    case 4:
    case 5:
        return 9;
    default:
        return 10;
    }
}

[[nodiscard]] std::pair<LoadedArcs, LoadedArcs> get_full_arcs(const mpi_rank_type rank) {
    const auto number_local_nodes = get_full_node_count(rank);

    // As in get_full_four_rank_graph(), every node is connected to every other node of the graph
    auto arcs = LoadedArcs(number_local_nodes);
    for (auto local_node = node_id_type{ 0 }; local_node < number_local_nodes; ++local_node) {
        for (auto other_rank = mpi_rank_type{ 0 }; other_rank < 7; ++other_rank) {
            const auto number_other_nodes = get_full_node_count(other_rank);
            for (auto other_node = node_id_type{ 0 }; other_node < number_other_nodes; ++other_node) {
                if (other_rank == rank && other_node == local_node) {
                    continue;
                }
                arcs[local_node][{ other_rank, other_node }] = 1;
            }
        }
    }

    auto in_arcs = arcs;
    return { std::move(in_arcs), std::move(arcs) };
}

[[nodiscard]] node_id_type merge_name(std::vector<std::string>& names, const std::string& name) {
    for (auto index = node_id_type{ 0 }; index < names.size(); ++index) {
        if (names[index] == name) {
            return index;
        }
    }

    names.push_back(name);
    return static_cast<node_id_type>(names.size() - 1);
}

void append_nodes(LoadedNodes& combined_nodes, const LoadedNodes& local_nodes) {
    auto area_mapping = std::vector<node_id_type>{};
    area_mapping.reserve(local_nodes.area_names.size());
    for (const auto& name : local_nodes.area_names) {
        area_mapping.push_back(merge_name(combined_nodes.area_names, name));
    }

    auto signal_mapping = std::vector<node_id_type>{};
    signal_mapping.reserve(local_nodes.signal_types.size());
    for (const auto& name : local_nodes.signal_types) {
        signal_mapping.push_back(merge_name(combined_nodes.signal_types, name));
    }

    combined_nodes.positions.insert(combined_nodes.positions.end(), local_nodes.positions.begin(), local_nodes.positions.end());
    for (const auto local_index : local_nodes.area_names_ind) {
        combined_nodes.area_names_ind.push_back(area_mapping[local_index]);
    }
    for (const auto local_index : local_nodes.signal_types_ind) {
        combined_nodes.signal_types_ind.push_back(signal_mapping[local_index]);
    }
}

[[nodiscard]] std::vector<node_id_type> get_node_prefixes(const bool full) {
    auto prefixes = std::vector<node_id_type>(8, 0);
    for (auto rank = mpi_rank_type{ 0 }; rank < 7; ++rank) {
        const auto number_nodes = full ? get_full_node_count(rank) : number_standard_nodes;
        prefixes[static_cast<std::size_t>(rank + 1)] = prefixes[static_cast<std::size_t>(rank)] + number_nodes;
    }
    return prefixes;
}

void append_arcs(LoadedArcs& combined_in_arcs, LoadedArcs& combined_out_arcs, const mpi_rank_type rank,
                 const std::vector<node_id_type>& prefixes, const LoadedArcs& local_in_arcs, const LoadedArcs& local_out_arcs) {
    const auto local_prefix = prefixes[static_cast<std::size_t>(rank)];

    for (auto local_node = node_id_type{ 0 }; local_node < local_in_arcs.size(); ++local_node) {
        const auto combined_node = local_prefix + local_node;
        for (const auto& [source_identifier, weight] : local_in_arcs[local_node]) {
            const auto& [source_rank, source_node] = source_identifier;
            const auto source = prefixes[static_cast<std::size_t>(source_rank)] + source_node;
            combined_in_arcs[combined_node][{ 0, source }] = weight;
        }
        for (const auto& [target_identifier, weight] : local_out_arcs[local_node]) {
            const auto& [target_rank, target_node] = target_identifier;
            const auto target = prefixes[static_cast<std::size_t>(target_rank)] + target_node;
            combined_out_arcs[combined_node][{ 0, target }] = weight;
        }
    }
}

[[nodiscard]] std::pair<LoadedArcs, LoadedArcs> get_rank_standard_arcs(const mpi_rank_type rank) {
    const auto assignment = get_rank_assignment(rank);
    return rank == 3 ? get_source_sink_arcs(number_standard_nodes)
                     : get_standard_arcs(rank, assignment.predecessor, assignment.successor);
}

[[nodiscard]] DistributedGraph make_standard_seven_rank_graph_on_one_rank(const bool undirected_unit_weight) {
    const auto prefixes = get_node_prefixes(false);
    auto loaded_nodes = LoadedNodes{};
    auto in_arcs = LoadedArcs(prefixes.back());
    auto out_arcs = LoadedArcs(prefixes.back());

    for (auto rank = mpi_rank_type{ 0 }; rank < 7; ++rank) {
        append_nodes(loaded_nodes, get_nodes_of_rank(rank, false));
        const auto [local_in_arcs, local_out_arcs] = get_rank_standard_arcs(rank);
        append_arcs(in_arcs, out_arcs, rank, prefixes, local_in_arcs, local_out_arcs);
    }

    if (!undirected_unit_weight) {
        return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
    }

    auto synchronized_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));
    auto unit_arcs = ArcTransformer::all_weights_one(std::move(synchronized_arcs));
    return DistributedGraph::construct_graph(std::move(loaded_nodes), unit_arcs, unit_arcs);
}
} // namespace

DistributedGraph GraphTest::get_standard_seven_rank_graph() {
    return make_standard_seven_rank_graph(false);
}

DistributedGraph GraphTest::get_standard_uu_seven_rank_graph() {
    return make_standard_seven_rank_graph(true);
}

DistributedGraph GraphTest::get_full_seven_rank_graph() {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    auto loaded_nodes = get_nodes_of_rank(my_rank, true);

    const auto [in_arcs, out_arcs] = get_full_arcs(my_rank);
    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_seven_rank_graph_on_one_rank() {
    return make_standard_seven_rank_graph_on_one_rank(false);
}

DistributedGraph GraphTest::get_standard_uu_seven_rank_graph_on_one_rank() {
    return make_standard_seven_rank_graph_on_one_rank(true);
}

DistributedGraph GraphTest::get_full_seven_rank_graph_on_one_rank() {
    const auto prefixes = get_node_prefixes(true);
    auto loaded_nodes = LoadedNodes{};
    auto in_arcs = LoadedArcs(prefixes.back());
    auto out_arcs = LoadedArcs(prefixes.back());

    for (auto rank = mpi_rank_type{ 0 }; rank < 7; ++rank) {
        append_nodes(loaded_nodes, get_nodes_of_rank(rank, true));
        const auto [local_in_arcs, local_out_arcs] = get_full_arcs(rank);
        append_arcs(in_arcs, out_arcs, rank, prefixes, local_in_arcs, local_out_arcs);
    }

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}
