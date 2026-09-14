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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

DistributedGraph GraphTest::get_standard_one_rank_graph() {
    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);
    positions.emplace_back(8.2, 4.8, 6.6);
    positions.emplace_back(7.4, 4.4, 3.4);

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;

    constexpr auto construct_element = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 2), construct_element(0, 4, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 7, 1), construct_element(0, 4, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 7, 2), construct_element(0, 8, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 9, 1), construct_element(0, 2, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 3, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 4, 2), construct_element(0, 1, 3), construct_element(0, 5, 9),
                                         construct_element(0, 6, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 5, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 4, 1), construct_element(0, 0, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 2), construct_element(0, 0, 1) });

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 2), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 6, 3) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 4, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 5, 2), construct_element(0, 7, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 2, 2), construct_element(0, 6, 2), construct_element(0, 8, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 6, 9), construct_element(0, 7, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 2), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 3, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 2), construct_element(0, 4, 1) });

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_empty_one_rank_graph() {
    // A graph without any nodes and therefore without any arcs. The category label lists are kept non-empty
    // (they exist independently of the nodes), but no node references them.
    auto positions = std::vector<Vec3d>{};

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");

    auto area_names_indices = std::vector<node_id_type>{};
    auto signal_types_indices = std::vector<node_id_type>{};

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    const auto in_arcs = LoadedArcs{};
    const auto out_arcs = LoadedArcs{};

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_uu_one_rank_graph() {
    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);
    positions.emplace_back(8.2, 4.8, 6.6);
    positions.emplace_back(7.4, 4.4, 3.4);

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;

    constexpr auto construct_element = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 2), construct_element(0, 4, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 7, 1), construct_element(0, 4, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 7, 2), construct_element(0, 8, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 9, 1), construct_element(0, 2, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 3, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 4, 2), construct_element(0, 1, 3), construct_element(0, 5, 9),
                                         construct_element(0, 6, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 5, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 4, 1), construct_element(0, 0, 2) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 2), construct_element(0, 0, 1) });

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 2), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 6, 3) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 4, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 5, 2), construct_element(0, 7, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 2, 2), construct_element(0, 6, 2), construct_element(0, 8, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 6, 9), construct_element(0, 7, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 2), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 3, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 2), construct_element(0, 4, 1) });

    auto transformed_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));
    auto clamped_arcs = ArcTransformer::all_weights_one(std::move(transformed_arcs));

    return DistributedGraph::construct_graph(std::move(loaded_nodes), clamped_arcs, clamped_arcs);
}

DistributedGraph GraphTest::get_full_one_rank_graph() {
    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);
    positions.emplace_back(8.2, 4.8, 6.6);
    positions.emplace_back(7.4, 4.4, 3.4);

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;

    auto in_arcs = LoadedArcs{ 10, arc_collection{} };
    auto out_arcs = LoadedArcs{ 10, arc_collection{} };

    for (auto source_node_id = node_id_type{ 0 }; source_node_id < 10; ++source_node_id) {
        for (auto target_node_id = node_id_type{ 0 }; target_node_id < 10; ++target_node_id) {
            in_arcs[target_node_id][{ 0, source_node_id }] = 1;
            out_arcs[source_node_id][{ 0, target_node_id }] = 1;
        }
    }

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_negative_weight_one_rank_graph() {
    // A small graph whose arcs mostly carry negative weights, for testing algorithms that use the
    // weights by absolute value. The arcs are 0 -> 1 (-3), 0 -> 2 (2), 1 -> 2 (-1), 1 -> 3 (2),
    // and 2 -> 3 (-4).
    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(1.0, 0.0, 0.0);
    positions.emplace_back(0.0, 1.0, 0.0);
    positions.emplace_back(0.0, 0.0, 1.0);
    positions.emplace_back(1.0, 1.0, 1.0);

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;

    constexpr auto construct_element = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{});
    in_arcs.emplace_back(arc_collection{ construct_element(0, 0, -3) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 0, 2), construct_element(0, 1, -1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 1, 2), construct_element(0, 2, -4) });

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, -3), construct_element(0, 2, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 2, -1), construct_element(0, 3, 2) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, -4) });
    out_arcs.emplace_back(arc_collection{});

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_four_rank_graph_on_one_rank() {
    auto positions = std::vector<Vec3d>{};
    positions.reserve(40);

    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);
    positions.emplace_back(8.2, 4.8, 6.6);
    positions.emplace_back(7.4, 4.4, 3.4);

    positions.emplace_back(0.40, 0.31, 0.64);
    positions.emplace_back(0.90, 0.83, 0.47);
    positions.emplace_back(0.78, 0.45, 0.45);
    positions.emplace_back(0.41, 0.22, 0.02);
    positions.emplace_back(0.19, 0.93, 0.04);
    positions.emplace_back(0.21, 0.28, 0.84);
    positions.emplace_back(0.00, 0.41, 0.88);
    positions.emplace_back(0.12, 0.55, 0.02);
    positions.emplace_back(0.22, 0.91, 0.19);
    positions.emplace_back(0.43, 0.99, 0.61);

    positions.emplace_back(0.18, 0.26, 1.00);
    positions.emplace_back(0.72, 0.59, 0.06);
    positions.emplace_back(0.46, 0.67, 0.75);
    positions.emplace_back(0.56, 0.44, 0.46);
    positions.emplace_back(0.48, 0.12, 0.64);
    positions.emplace_back(0.17, 0.79, 0.01);
    positions.emplace_back(0.65, 0.17, 1.00);
    positions.emplace_back(0.58, 0.87, 0.38);
    positions.emplace_back(0.05, 0.57, 0.33);
    positions.emplace_back(0.96, 0.44, 0.95);

    positions.emplace_back(0.22, 0.29, 0.89);
    positions.emplace_back(0.99, 0.51, 0.16);
    positions.emplace_back(0.99, 0.72, 0.75);
    positions.emplace_back(0.71, 0.01, 0.71);
    positions.emplace_back(0.93, 0.68, 0.42);
    positions.emplace_back(0.98, 0.47, 0.17);
    positions.emplace_back(0.73, 0.48, 0.93);
    positions.emplace_back(0.91, 0.35, 0.43);
    positions.emplace_back(0.28, 0.12, 0.09);
    positions.emplace_back(0.08, 0.67, 0.17);

    auto area_names = std::vector<std::string>{};
    area_names.reserve(5);

    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");
    area_names.emplace_back("area_4");
    area_names.emplace_back("area_5");

    auto signal_types = std::vector<std::string>{};
    signal_types.reserve(2);

    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.reserve(40);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);

    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.reserve(40);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);

    using ac = LoadedArcs::value_type;

    constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.reserve(40);

    for (const auto mpi_rank : { 0U, 1U, 2U, 3U }) {
        const auto my_offset = 10U * mpi_rank;

        in_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 9 + my_offset, 2), ce(0, 4 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 7 + my_offset, 1), ce(0, 4 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 2 + my_offset, 1), ce(0, 7 + my_offset, 2), ce(0, 8 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 9 + my_offset, 1), ce(0, 2 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 5 + my_offset, 1), ce(0, 3 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 0 + my_offset, 1), ce(0, 4 + my_offset, 2), ce(0, 5 + my_offset, 9), ce(0, 1 + my_offset, 3), ce(0, 6 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 5 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 4 + my_offset, 1), ce(0, 0 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 1), ce(0, 8 + my_offset, 2), ce(0, 0 + my_offset, 1) });

        const auto other_rank = (mpi_rank + 3) % 4;
        const auto other_offset = 10 * other_rank;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            in_arcs[node_id + my_offset][{ 0, node_id + other_offset }] = 11;
        }
    }

    auto out_arcs = LoadedArcs{};
    out_arcs.reserve(40);

    for (const auto mpi_rank : { 0U, 1U, 2U, 3U }) {
        const auto my_offset = 10U * mpi_rank;

        out_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 1), ce(0, 8 + my_offset, 2), ce(0, 9 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 0 + my_offset, 1), ce(0, 6 + my_offset, 3) });
        out_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 4 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1), ce(0, 5 + my_offset, 2), ce(0, 7 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1), ce(0, 2 + my_offset, 2), ce(0, 6 + my_offset, 2), ce(0, 8 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 5 + my_offset, 1), ce(0, 6 + my_offset, 9), ce(0, 7 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 2), ce(0, 9 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 2 + my_offset, 1), ce(0, 3 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 9 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 2), ce(0, 4 + my_offset, 1) });

        const auto other_rank = (mpi_rank + 1) % 4;
        const auto other_offset = 10 * other_rank;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            out_arcs[node_id + my_offset][{ 0, node_id + other_offset }] = 11;
        }
    }

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_uu_four_rank_graph_on_one_rank() {
    auto positions = std::vector<Vec3d>{};
    positions.reserve(40);

    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);
    positions.emplace_back(8.2, 4.8, 6.6);
    positions.emplace_back(7.4, 4.4, 3.4);

    positions.emplace_back(0.40, 0.31, 0.64);
    positions.emplace_back(0.90, 0.83, 0.47);
    positions.emplace_back(0.78, 0.45, 0.45);
    positions.emplace_back(0.41, 0.22, 0.02);
    positions.emplace_back(0.19, 0.93, 0.04);
    positions.emplace_back(0.21, 0.28, 0.84);
    positions.emplace_back(0.00, 0.41, 0.88);
    positions.emplace_back(0.12, 0.55, 0.02);
    positions.emplace_back(0.22, 0.91, 0.19);
    positions.emplace_back(0.43, 0.99, 0.61);

    positions.emplace_back(0.18, 0.26, 1.00);
    positions.emplace_back(0.72, 0.59, 0.06);
    positions.emplace_back(0.46, 0.67, 0.75);
    positions.emplace_back(0.56, 0.44, 0.46);
    positions.emplace_back(0.48, 0.12, 0.64);
    positions.emplace_back(0.17, 0.79, 0.01);
    positions.emplace_back(0.65, 0.17, 1.00);
    positions.emplace_back(0.58, 0.87, 0.38);
    positions.emplace_back(0.05, 0.57, 0.33);
    positions.emplace_back(0.96, 0.44, 0.95);

    positions.emplace_back(0.22, 0.29, 0.89);
    positions.emplace_back(0.99, 0.51, 0.16);
    positions.emplace_back(0.99, 0.72, 0.75);
    positions.emplace_back(0.71, 0.01, 0.71);
    positions.emplace_back(0.93, 0.68, 0.42);
    positions.emplace_back(0.98, 0.47, 0.17);
    positions.emplace_back(0.73, 0.48, 0.93);
    positions.emplace_back(0.91, 0.35, 0.43);
    positions.emplace_back(0.28, 0.12, 0.09);
    positions.emplace_back(0.08, 0.67, 0.17);

    auto area_names = std::vector<std::string>{};
    area_names.reserve(5);

    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");
    area_names.emplace_back("area_4");
    area_names.emplace_back("area_5");

    auto signal_types = std::vector<std::string>{};
    signal_types.reserve(2);

    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.reserve(40);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);

    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.reserve(40);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);

    using ac = LoadedArcs::value_type;

    constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.reserve(40);

    for (const auto mpi_rank : { 0U, 1U, 2U, 3U }) {
        const auto my_offset = 10U * mpi_rank;

        in_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 9 + my_offset, 2), ce(0, 4 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 7 + my_offset, 1), ce(0, 4 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 2 + my_offset, 1), ce(0, 7 + my_offset, 2), ce(0, 8 + my_offset, 1) });
        in_arcs.emplace_back(ac{ ce(0, 9 + my_offset, 1), ce(0, 2 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 5 + my_offset, 1), ce(0, 3 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 0 + my_offset, 1), ce(0, 4 + my_offset, 2), ce(0, 5 + my_offset, 9), ce(0, 1 + my_offset, 3), ce(0, 6 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 5 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 4 + my_offset, 1), ce(0, 0 + my_offset, 2) });
        in_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 1), ce(0, 8 + my_offset, 2), ce(0, 0 + my_offset, 1) });

        const auto other_rank = (mpi_rank + 3) % 4;
        const auto other_offset = 10 * other_rank;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            in_arcs[node_id + my_offset][{ 0, node_id + other_offset }] = 11;
        }
    }

    auto out_arcs = LoadedArcs{};
    out_arcs.reserve(40);

    for (const auto mpi_rank : { 0U, 1U, 2U, 3U }) {
        const auto my_offset = 10U * mpi_rank;

        out_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 1), ce(0, 8 + my_offset, 2), ce(0, 9 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 0 + my_offset, 1), ce(0, 6 + my_offset, 3) });
        out_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 4 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1), ce(0, 5 + my_offset, 2), ce(0, 7 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 1), ce(0, 2 + my_offset, 2), ce(0, 6 + my_offset, 2), ce(0, 8 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 5 + my_offset, 1), ce(0, 6 + my_offset, 9), ce(0, 7 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 6 + my_offset, 2), ce(0, 9 + my_offset, 1) });
        out_arcs.emplace_back(ac{ ce(0, 2 + my_offset, 1), ce(0, 3 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 3 + my_offset, 1), ce(0, 9 + my_offset, 2) });
        out_arcs.emplace_back(ac{ ce(0, 1 + my_offset, 2), ce(0, 4 + my_offset, 1) });

        const auto other_rank = (mpi_rank + 1) % 4;
        const auto other_offset = 10 * other_rank;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            out_arcs[node_id + my_offset][{ 0, node_id + other_offset }] = 11;
        }
    }

    auto transformed_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));
    auto clamped_arcs = ArcTransformer::all_weights_one(std::move(transformed_arcs));

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    return DistributedGraph::construct_graph(std::move(loaded_nodes), clamped_arcs, clamped_arcs);
}

DistributedGraph GraphTest::get_full_four_rank_graph_on_one_rank() {
    auto positions = std::vector<Vec3d>{};
    positions.reserve(36);

    positions.emplace_back(0.1, 9.8, 4.7);
    positions.emplace_back(9.6, 5.7, 1.4);
    positions.emplace_back(0.3, 3.9, 8.0);
    positions.emplace_back(7.9, 2.8, 9.6);
    positions.emplace_back(2.2, 4.9, 0.0);
    positions.emplace_back(0.9, 5.2, 8.6);
    positions.emplace_back(0.6, 5.8, 5.2);
    positions.emplace_back(8.2, 2.0, 8.8);

    positions.emplace_back(0.40, 0.31, 0.64);
    positions.emplace_back(0.90, 0.83, 0.47);
    positions.emplace_back(0.78, 0.45, 0.45);
    positions.emplace_back(0.41, 0.22, 0.02);
    positions.emplace_back(0.19, 0.93, 0.04);
    positions.emplace_back(0.21, 0.28, 0.84);
    positions.emplace_back(0.00, 0.41, 0.88);
    positions.emplace_back(0.12, 0.55, 0.02);
    positions.emplace_back(0.22, 0.91, 0.19);

    positions.emplace_back(0.18, 0.26, 1.00);
    positions.emplace_back(0.72, 0.59, 0.06);
    positions.emplace_back(0.46, 0.67, 0.75);
    positions.emplace_back(0.56, 0.44, 0.46);
    positions.emplace_back(0.48, 0.12, 0.64);
    positions.emplace_back(0.17, 0.79, 0.01);
    positions.emplace_back(0.65, 0.17, 1.00);
    positions.emplace_back(0.58, 0.87, 0.38);
    positions.emplace_back(0.05, 0.57, 0.33);

    positions.emplace_back(0.22, 0.29, 0.89);
    positions.emplace_back(0.99, 0.51, 0.16);
    positions.emplace_back(0.99, 0.72, 0.75);
    positions.emplace_back(0.71, 0.01, 0.71);
    positions.emplace_back(0.93, 0.68, 0.42);
    positions.emplace_back(0.98, 0.47, 0.17);
    positions.emplace_back(0.73, 0.48, 0.93);
    positions.emplace_back(0.91, 0.35, 0.43);
    positions.emplace_back(0.28, 0.12, 0.09);
    positions.emplace_back(0.08, 0.67, 0.17);

    auto area_names = std::vector<std::string>{};
    area_names.reserve(5);

    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");
    area_names.emplace_back("area_4");
    area_names.emplace_back("area_5");

    auto signal_types = std::vector<std::string>{};
    signal_types.reserve(2);

    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    auto area_names_indices = std::vector<node_id_type>{};
    area_names_indices.reserve(36);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(2);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);

    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(0);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);
    area_names_indices.emplace_back(4);

    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(1);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(3);
    area_names_indices.emplace_back(1);

    auto signal_types_indices = std::vector<node_id_type>{};
    signal_types_indices.reserve(36);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);

    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(1);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(0);
    signal_types_indices.emplace_back(1);

    auto in_arcs = LoadedArcs{ 36, LoadedArcs::value_type{} };
    auto out_arcs = LoadedArcs{ 36, LoadedArcs::value_type{} };

    for (auto my_node_id = node_id_type{ 0 }; my_node_id < 36; ++my_node_id) {
        for (auto other_node_id = node_id_type{ 0 }; other_node_id < 36; ++other_node_id) {
            if (my_node_id == other_node_id) {
                continue;
            }

            in_arcs[my_node_id][{ 0, other_node_id }] = 1;
            out_arcs[my_node_id][{ 0, other_node_id }] = 1;
        }
    }

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}
