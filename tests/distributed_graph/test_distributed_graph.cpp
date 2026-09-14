/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_distributed_graph.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

template <std::equality_comparable T>
[[nodiscard]] constexpr bool operator==(const std::span<T>& lhs, const std::span<T>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    return std::ranges::equal(lhs, rhs);
}

TEST_F(DistributedGraphTest, testLoadNodes) {
    if (skip_unless_rank_count(1)) {
        return;
    }

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

    auto graph = DistributedGraph::construct_graph(std::move(loaded_nodes), {}, {});

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 0);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 0);

    const auto my_rank = mpi_rank_type{ 0 };

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id), arc_id_type{ 0 });
        ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id), arc_id_type{ 0 });

        ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id), arc_id_type{ 0 });
        ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id), arc_id_type{ 0 });

        ASSERT_EQ(graph.get_in_arcs(my_rank, node_id).size(), 0);
        ASSERT_EQ(graph.get_out_arcs(my_rank, node_id).size(), 0);
    }

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 0 }), Vec3d(0.1, 9.8, 4.7));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 1 }), Vec3d(9.6, 5.7, 1.4));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 2 }), Vec3d(0.3, 3.9, 8.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 3 }), Vec3d(7.9, 2.8, 9.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 4 }), Vec3d(2.2, 4.9, 0.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 5 }), Vec3d(0.9, 5.2, 8.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 6 }), Vec3d(0.6, 5.8, 5.2));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 7 }), Vec3d(8.2, 2.0, 8.8));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 8 }), Vec3d(8.2, 4.8, 6.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 9 }), Vec3d(7.4, 4.4, 3.4));

    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 4 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 5 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 6 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 7 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 8 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 9 }), 2);

    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 1 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 4 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 5 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 6 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 7 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 8 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 9 }), 0);
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcs) {
    if (skip_unless_rank_count(1)) {
        return;
    }

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

    using arc_collection = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>;

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

    auto graph = DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 25);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 25);

    const auto my_rank = mpi_rank_type{ 0 };

    const auto check_in_arcs = [&graph, &in_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id), in_arcs[node_id].size());
        ASSERT_EQ(graph.get_in_arcs(my_rank, node_id).size(), in_arcs[node_id].size());

        auto graph_set = std::set<InArc>{};
        graph_set.insert(graph.get_in_arcs(my_rank, node_id).begin(), graph.get_in_arcs(my_rank, node_id).end());

        auto arc_set = std::set<InArc>{};
        for (const auto& [key, weight] : in_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set);

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_in_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_in_arcs(my_rank, node_id)[arc_id], graph.get_in_arc(my_rank, node_id, arc_id));
        }
    };

    const auto check_out_arcs = [&graph, &out_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id), out_arcs[node_id].size());
        ASSERT_EQ(graph.get_out_arcs(my_rank, node_id).size(), out_arcs[node_id].size());

        auto graph_set = std::set<OutArc>{};
        graph_set.insert(graph.get_out_arcs(my_rank, node_id).begin(), graph.get_out_arcs(my_rank, node_id).end());

        auto arc_set = std::set<OutArc>{};
        for (const auto& [key, weight] : out_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set);

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_out_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_out_arcs(my_rank, node_id)[arc_id], graph.get_out_arc(my_rank, node_id, arc_id));
        }
    };

    const auto in_arc_info_distribution = graph.get_local_in_arc_info_distribution();
    const auto out_arc_info_distribution = graph.get_local_out_arc_info_distribution();

    ASSERT_EQ(in_arc_info_distribution.size(), node_id_type{ 10 });
    ASSERT_EQ(out_arc_info_distribution.size(), node_id_type{ 10 });

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        ASSERT_EQ(in_arc_info_distribution[node_id], graph.get_in_arc_info(0, node_id));
        ASSERT_EQ(out_arc_info_distribution[node_id], graph.get_out_arc_info(0, node_id));
    }

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        check_in_arcs(node_id);
        check_out_arcs(node_id);
    }

    const auto weight_in_arcs = graph.get_weight_in_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 1, 4, 3, 4, 3, 3, 17, 3, 3, 4 };
        ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_in_arcs[node_id], expected_weights[node_id]);
    }

    const auto weight_out_arcs = graph.get_weight_out_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 4, 4, 3, 4, 6, 12, 3, 3, 3, 3 };
        ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_out_arcs[node_id], expected_weights[node_id]);
    }

    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 2 }), 4);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 3 }), 6);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 4 }), 9);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 5 }), 11);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 6 }), 13);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 7 }), 18);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 8 }), 20);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 9 }), 22);

    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 1 }), 3);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 2 }), 5);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 3 }), 7);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 4 }), 10);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 5 }), 14);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 6 }), 17);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 7 }), 19);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 8 }), 21);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 9 }), 23);

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 0 }), Vec3d(0.1, 9.8, 4.7));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 1 }), Vec3d(9.6, 5.7, 1.4));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 2 }), Vec3d(0.3, 3.9, 8.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 3 }), Vec3d(7.9, 2.8, 9.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 4 }), Vec3d(2.2, 4.9, 0.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 5 }), Vec3d(0.9, 5.2, 8.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 6 }), Vec3d(0.6, 5.8, 5.2));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 7 }), Vec3d(8.2, 2.0, 8.8));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 8 }), Vec3d(8.2, 4.8, 6.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 9 }), Vec3d(7.4, 4.4, 3.4));

    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 4 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 5 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 6 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 7 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 8 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 9 }), 2);

    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 1 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 4 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 5 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 6 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 7 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 8 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 9 }), 0);
}

TEST_F(DistributedGraphTest, testLoadNodesFile) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
    auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
    auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };

    auto write_to_nodes_file = [&nodes_file](const node_id_type node_id, const double x, const double y, const double z, const std::string_view area,
                                             const std::string_view signal_type) {
        fmt::print(nodes_file, "{} {} {} {} {} {}\n", node_id, x, y, z, area, signal_type);
    };

    write_to_nodes_file(1, 0.1, 9.8, 4.7, "area_1", "excitatory");
    write_to_nodes_file(2, 9.6, 5.7, 1.4, "area_2", "excitatory");
    write_to_nodes_file(3, 0.3, 3.9, 8.0, "area_2", "inhibitory");
    write_to_nodes_file(4, 7.9, 2.8, 9.6, "area_1", "excitatory");
    write_to_nodes_file(5, 2.2, 4.9, 0.0, "area_2", "excitatory");
    write_to_nodes_file(6, 0.9, 5.2, 8.6, "area_2", "excitatory");
    write_to_nodes_file(7, 0.6, 5.8, 5.2, "area_1", "inhibitory");
    write_to_nodes_file(8, 8.2, 2.0, 8.8, "area_2", "excitatory");
    write_to_nodes_file(9, 8.2, 4.8, 6.6, "area_1", "inhibitory");
    write_to_nodes_file(10, 7.4, 4.4, 3.4, "area_3", "excitatory");

    nodes_file.flush();
    nodes_file.close();

    in_arcs_file.flush();
    in_arcs_file.close();

    out_arcs_file.flush();
    out_arcs_file.close();

    auto graph = DistributedGraph::construct_graph("./input/", false, false, false, "");

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 0);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 0);

    const auto my_rank = mpi_rank_type{ 0 };

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id), arc_id_type{ 0 });
        ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id), arc_id_type{ 0 });

        ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id), arc_id_type{ 0 });
        ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id), arc_id_type{ 0 });

        ASSERT_EQ(graph.get_in_arcs(my_rank, node_id).size(), 0);
        ASSERT_EQ(graph.get_out_arcs(my_rank, node_id).size(), 0);
    }

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 0 }), Vec3d(0.1, 9.8, 4.7));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 1 }), Vec3d(9.6, 5.7, 1.4));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 2 }), Vec3d(0.3, 3.9, 8.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 3 }), Vec3d(7.9, 2.8, 9.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 4 }), Vec3d(2.2, 4.9, 0.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 5 }), Vec3d(0.9, 5.2, 8.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 6 }), Vec3d(0.6, 5.8, 5.2));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 7 }), Vec3d(8.2, 2.0, 8.8));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 8 }), Vec3d(8.2, 4.8, 6.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 9 }), Vec3d(7.4, 4.4, 3.4));

    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 4 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 5 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 6 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 7 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 8 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 9 }), 2);

    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 1 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 4 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 5 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 6 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 7 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 8 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 9 }), 0);
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcsFile) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
    auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
    auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };

    auto write_to_nodes_file = [&nodes_file](const node_id_type node_id, const double x, const double y, const double z, const std::string& area,
                                             const std::string& signal_type) {
        fmt::print(nodes_file, "{} {} {} {} {} {}\n", node_id, x, y, z, area, signal_type);
    };

    auto write_to_in_arcs_file = [&in_arcs_file](const mpi_rank_type target_rank, const node_id_type target_node_id, const mpi_rank_type source_rank,
                                                 const node_id_type source_node_id, const weight_type weight, const bool plastic) {
        fmt::print(in_arcs_file, "{} {} {} {} {} {}\n", target_rank, target_node_id, source_rank, source_node_id, weight, plastic);
    };

    auto write_to_out_arcs_file = [&out_arcs_file](const mpi_rank_type target_rank, const node_id_type target_node_id, const mpi_rank_type source_rank,
                                                   const node_id_type source_node_id, const weight_type weight, const bool plastic) {
        fmt::print(out_arcs_file, "{} {} {} {} {} {}\n", target_rank, target_node_id, source_rank, source_node_id, weight, plastic);
    };

    write_to_nodes_file(1, 0.1, 9.8, 4.7, "area_1", "excitatory");
    write_to_nodes_file(2, 9.6, 5.7, 1.4, "area_2", "excitatory");
    write_to_nodes_file(3, 0.3, 3.9, 8.0, "area_2", "inhibitory");
    write_to_nodes_file(4, 7.9, 2.8, 9.6, "area_1", "excitatory");
    write_to_nodes_file(5, 2.2, 4.9, 0.0, "area_2", "excitatory");
    write_to_nodes_file(6, 0.9, 5.2, 8.6, "area_2", "excitatory");
    write_to_nodes_file(7, 0.6, 5.8, 5.2, "area_1", "inhibitory");
    write_to_nodes_file(8, 8.2, 2.0, 8.8, "area_2", "excitatory");
    write_to_nodes_file(9, 8.2, 4.8, 6.6, "area_1", "inhibitory");
    write_to_nodes_file(10, 7.4, 4.4, 3.4, "area_3", "excitatory");

    write_to_in_arcs_file(0, 1, 0, 2, 1, true);
    write_to_in_arcs_file(0, 2, 0, 4, 1, true);
    write_to_in_arcs_file(0, 2, 0, 10, 2, true);
    write_to_in_arcs_file(0, 2, 0, 5, 1, true);
    write_to_in_arcs_file(0, 3, 0, 8, 1, true);
    write_to_in_arcs_file(0, 3, 0, 5, 2, true);
    write_to_in_arcs_file(0, 4, 0, 3, 1, true);
    write_to_in_arcs_file(0, 4, 0, 8, 2, true);
    write_to_in_arcs_file(0, 4, 0, 9, 1, true);
    write_to_in_arcs_file(0, 5, 0, 10, 1, true);
    write_to_in_arcs_file(0, 5, 0, 3, 2, true);
    write_to_in_arcs_file(0, 6, 0, 6, 1, true);
    write_to_in_arcs_file(0, 6, 0, 4, 2, true);
    write_to_in_arcs_file(0, 7, 0, 1, 1, true);
    write_to_in_arcs_file(0, 7, 0, 2, 3, true);
    write_to_in_arcs_file(0, 7, 0, 5, 2, true);
    write_to_in_arcs_file(0, 7, 0, 6, 9, true);
    write_to_in_arcs_file(0, 7, 0, 7, 2, true);
    write_to_in_arcs_file(0, 8, 0, 4, 1, true);
    write_to_in_arcs_file(0, 8, 0, 6, 2, true);
    write_to_in_arcs_file(0, 9, 0, 5, 1, true);
    write_to_in_arcs_file(0, 9, 0, 1, 2, true);
    write_to_in_arcs_file(0, 10, 0, 7, 1, true);
    write_to_in_arcs_file(0, 10, 0, 9, 2, true);
    write_to_in_arcs_file(0, 10, 0, 1, 1, true);

    write_to_out_arcs_file(0, 1, 0, 2, 1, true);
    write_to_out_arcs_file(0, 2, 0, 4, 1, true);
    write_to_out_arcs_file(0, 2, 0, 10, 2, true);
    write_to_out_arcs_file(0, 2, 0, 5, 1, true);
    write_to_out_arcs_file(0, 3, 0, 8, 1, true);
    write_to_out_arcs_file(0, 3, 0, 5, 2, true);
    write_to_out_arcs_file(0, 4, 0, 3, 1, true);
    write_to_out_arcs_file(0, 4, 0, 8, 2, true);
    write_to_out_arcs_file(0, 4, 0, 9, 1, true);
    write_to_out_arcs_file(0, 5, 0, 10, 1, true);
    write_to_out_arcs_file(0, 5, 0, 3, 2, true);
    write_to_out_arcs_file(0, 6, 0, 6, 1, true);
    write_to_out_arcs_file(0, 6, 0, 4, 2, true);
    write_to_out_arcs_file(0, 7, 0, 1, 1, true);
    write_to_out_arcs_file(0, 7, 0, 2, 3, true);
    write_to_out_arcs_file(0, 7, 0, 5, 2, true);
    write_to_out_arcs_file(0, 7, 0, 6, 9, true);
    write_to_out_arcs_file(0, 7, 0, 7, 2, true);
    write_to_out_arcs_file(0, 8, 0, 4, 1, true);
    write_to_out_arcs_file(0, 8, 0, 6, 2, true);
    write_to_out_arcs_file(0, 9, 0, 5, 1, true);
    write_to_out_arcs_file(0, 9, 0, 1, 2, true);
    write_to_out_arcs_file(0, 10, 0, 7, 1, true);
    write_to_out_arcs_file(0, 10, 0, 9, 2, true);
    write_to_out_arcs_file(0, 10, 0, 1, 1, true);

    nodes_file.flush();
    nodes_file.close();

    in_arcs_file.flush();
    in_arcs_file.close();

    out_arcs_file.flush();
    out_arcs_file.close();

    auto graph = DistributedGraph::construct_graph("./input/", false, false, false, "");

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    using arc_collection = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>;

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

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 25);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 25);

    const auto my_rank = mpi_rank_type{ 0 };

    const auto check_in_arcs = [&graph, &in_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id), in_arcs[node_id].size()) << ' ' << node_id;
        ASSERT_EQ(graph.get_in_arcs(my_rank, node_id).size(), in_arcs[node_id].size()) << ' ' << node_id;

        auto graph_set = std::set<InArc>{};
        graph_set.insert(graph.get_in_arcs(my_rank, node_id).begin(), graph.get_in_arcs(my_rank, node_id).end());

        auto arc_set = std::set<InArc>{};
        for (const auto& [key, weight] : in_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_in_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_in_arcs(my_rank, node_id)[arc_id], graph.get_in_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }
    };

    const auto check_out_arcs = [&graph, &out_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id), out_arcs[node_id].size()) << ' ' << node_id;
        ASSERT_EQ(graph.get_out_arcs(my_rank, node_id).size(), out_arcs[node_id].size()) << ' ' << node_id;

        auto graph_set = std::set<OutArc>{};
        graph_set.insert(graph.get_out_arcs(my_rank, node_id).begin(), graph.get_out_arcs(my_rank, node_id).end());

        auto arc_set = std::set<OutArc>{};
        for (const auto& [key, weight] : out_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_out_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_out_arcs(my_rank, node_id)[arc_id], graph.get_out_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }
    };

    const auto in_arc_info_distribution = graph.get_local_in_arc_info_distribution();
    const auto out_arc_info_distribution = graph.get_local_out_arc_info_distribution();

    ASSERT_EQ(in_arc_info_distribution.size(), node_id_type{ 10 });
    ASSERT_EQ(out_arc_info_distribution.size(), node_id_type{ 10 });

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        ASSERT_EQ(in_arc_info_distribution[node_id], graph.get_in_arc_info(0, node_id));
        ASSERT_EQ(out_arc_info_distribution[node_id], graph.get_out_arc_info(0, node_id));
    }

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        check_in_arcs(node_id);
        check_out_arcs(node_id);
    }

    const auto weight_in_arcs = graph.get_weight_in_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 1, 4, 3, 4, 3, 3, 17, 3, 3, 4 };
        ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_in_arcs[node_id], expected_weights[node_id]);
    }

    const auto weight_out_arcs = graph.get_weight_out_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 4, 4, 3, 4, 6, 12, 3, 3, 3, 3 };
        ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_out_arcs[node_id], expected_weights[node_id]);
    }

    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 2 }), 4);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 3 }), 6);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 4 }), 9);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 5 }), 11);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 6 }), 13);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 7 }), 18);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 8 }), 20);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 9 }), 22);

    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 1 }), 3);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 2 }), 5);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 3 }), 7);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 4 }), 10);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 5 }), 14);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 6 }), 17);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 7 }), 19);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 8 }), 21);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 9 }), 23);

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 0 }), Vec3d(0.1, 9.8, 4.7));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 1 }), Vec3d(9.6, 5.7, 1.4));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 2 }), Vec3d(0.3, 3.9, 8.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 3 }), Vec3d(7.9, 2.8, 9.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 4 }), Vec3d(2.2, 4.9, 0.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 5 }), Vec3d(0.9, 5.2, 8.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 6 }), Vec3d(0.6, 5.8, 5.2));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 7 }), Vec3d(8.2, 2.0, 8.8));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 8 }), Vec3d(8.2, 4.8, 6.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 9 }), Vec3d(7.4, 4.4, 3.4));

    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 4 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 5 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 6 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 7 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 8 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 9 }), 2);

    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 1 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 4 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 5 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 6 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 7 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 8 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 9 }), 0);
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcsFileWeightsOne) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
    auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
    auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };

    auto write_to_nodes_file = [&nodes_file](const node_id_type node_id, const double x, const double y, const double z, const std::string& area,
                                             const std::string& signal_type) {
        fmt::print(nodes_file, "{} {} {} {} {} {}\n", node_id, x, y, z, area, signal_type);
    };

    auto write_to_in_arcs_file = [&in_arcs_file](const mpi_rank_type target_rank, const node_id_type target_node_id, const mpi_rank_type source_rank,
                                                 const node_id_type source_node_id, const weight_type weight, const bool plastic) {
        fmt::print(in_arcs_file, "{} {} {} {} {} {}\n", target_rank, target_node_id, source_rank, source_node_id, weight, plastic);
    };

    auto write_to_out_arcs_file = [&out_arcs_file](const mpi_rank_type target_rank, const node_id_type target_node_id, const mpi_rank_type source_rank,
                                                   const node_id_type source_node_id, const weight_type weight, const bool plastic) {
        fmt::print(out_arcs_file, "{} {} {} {} {} {}\n", target_rank, target_node_id, source_rank, source_node_id, weight, plastic);
    };

    write_to_nodes_file(1, 0.1, 9.8, 4.7, "area_1", "excitatory");
    write_to_nodes_file(2, 9.6, 5.7, 1.4, "area_2", "excitatory");
    write_to_nodes_file(3, 0.3, 3.9, 8.0, "area_2", "inhibitory");
    write_to_nodes_file(4, 7.9, 2.8, 9.6, "area_1", "excitatory");
    write_to_nodes_file(5, 2.2, 4.9, 0.0, "area_2", "excitatory");
    write_to_nodes_file(6, 0.9, 5.2, 8.6, "area_2", "excitatory");
    write_to_nodes_file(7, 0.6, 5.8, 5.2, "area_1", "inhibitory");
    write_to_nodes_file(8, 8.2, 2.0, 8.8, "area_2", "excitatory");
    write_to_nodes_file(9, 8.2, 4.8, 6.6, "area_1", "inhibitory");
    write_to_nodes_file(10, 7.4, 4.4, 3.4, "area_3", "excitatory");

    write_to_in_arcs_file(0, 1, 0, 2, 1, true);
    write_to_in_arcs_file(0, 2, 0, 4, 1, true);
    write_to_in_arcs_file(0, 2, 0, 10, 2, true);
    write_to_in_arcs_file(0, 2, 0, 5, 1, true);
    write_to_in_arcs_file(0, 3, 0, 8, 1, true);
    write_to_in_arcs_file(0, 3, 0, 5, 2, true);
    write_to_in_arcs_file(0, 4, 0, 3, 1, true);
    write_to_in_arcs_file(0, 4, 0, 8, 2, true);
    write_to_in_arcs_file(0, 4, 0, 9, 1, true);
    write_to_in_arcs_file(0, 5, 0, 10, 1, true);
    write_to_in_arcs_file(0, 5, 0, 3, 2, true);
    write_to_in_arcs_file(0, 6, 0, 6, 1, true);
    write_to_in_arcs_file(0, 6, 0, 4, 2, true);
    write_to_in_arcs_file(0, 7, 0, 1, 1, true);
    write_to_in_arcs_file(0, 7, 0, 2, 3, true);
    write_to_in_arcs_file(0, 7, 0, 5, 2, true);
    write_to_in_arcs_file(0, 7, 0, 6, 9, true);
    write_to_in_arcs_file(0, 7, 0, 7, 2, true);
    write_to_in_arcs_file(0, 8, 0, 4, 1, true);
    write_to_in_arcs_file(0, 8, 0, 6, 2, true);
    write_to_in_arcs_file(0, 9, 0, 5, 1, true);
    write_to_in_arcs_file(0, 9, 0, 1, 2, true);
    write_to_in_arcs_file(0, 10, 0, 7, 1, true);
    write_to_in_arcs_file(0, 10, 0, 9, 2, true);
    write_to_in_arcs_file(0, 10, 0, 1, 1, true);

    write_to_out_arcs_file(0, 1, 0, 2, 1, true);
    write_to_out_arcs_file(0, 2, 0, 4, 1, true);
    write_to_out_arcs_file(0, 2, 0, 10, 2, true);
    write_to_out_arcs_file(0, 2, 0, 5, 1, true);
    write_to_out_arcs_file(0, 3, 0, 8, 1, true);
    write_to_out_arcs_file(0, 3, 0, 5, 2, true);
    write_to_out_arcs_file(0, 4, 0, 3, 1, true);
    write_to_out_arcs_file(0, 4, 0, 8, 2, true);
    write_to_out_arcs_file(0, 4, 0, 9, 1, true);
    write_to_out_arcs_file(0, 5, 0, 10, 1, true);
    write_to_out_arcs_file(0, 5, 0, 3, 2, true);
    write_to_out_arcs_file(0, 6, 0, 6, 1, true);
    write_to_out_arcs_file(0, 6, 0, 4, 2, true);
    write_to_out_arcs_file(0, 7, 0, 1, 1, true);
    write_to_out_arcs_file(0, 7, 0, 2, 3, true);
    write_to_out_arcs_file(0, 7, 0, 5, 2, true);
    write_to_out_arcs_file(0, 7, 0, 6, 9, true);
    write_to_out_arcs_file(0, 7, 0, 7, 2, true);
    write_to_out_arcs_file(0, 8, 0, 4, 1, true);
    write_to_out_arcs_file(0, 8, 0, 6, 2, true);
    write_to_out_arcs_file(0, 9, 0, 5, 1, true);
    write_to_out_arcs_file(0, 9, 0, 1, 2, true);
    write_to_out_arcs_file(0, 10, 0, 7, 1, true);
    write_to_out_arcs_file(0, 10, 0, 9, 2, true);
    write_to_out_arcs_file(0, 10, 0, 1, 1, true);

    nodes_file.flush();
    nodes_file.close();

    in_arcs_file.flush();
    in_arcs_file.close();

    out_arcs_file.flush();
    out_arcs_file.close();

    auto graph = DistributedGraph::construct_graph("./input/", false, false, true, "");

    auto area_names = std::vector<std::string>{};
    area_names.emplace_back("area_1");
    area_names.emplace_back("area_2");
    area_names.emplace_back("area_3");

    auto signal_types = std::vector<std::string>{};
    signal_types.emplace_back("excitatory");
    signal_types.emplace_back("inhibitory");

    using arc_collection = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>;

    constexpr auto construct_element = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 1), construct_element(0, 4, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 7, 1), construct_element(0, 4, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 7, 1), construct_element(0, 8, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 9, 1), construct_element(0, 2, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 3, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 4, 1), construct_element(0, 1, 1), construct_element(0, 5, 1),
                                         construct_element(0, 6, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 5, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 4, 1), construct_element(0, 0, 1) });
    in_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 1), construct_element(0, 0, 1) });

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 8, 1), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 0, 1), construct_element(0, 6, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 4, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 5, 1), construct_element(0, 7, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 2, 1), construct_element(0, 6, 1), construct_element(0, 8, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 5, 1), construct_element(0, 6, 1), construct_element(0, 7, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 6, 1), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 2, 1), construct_element(0, 3, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 3, 1), construct_element(0, 9, 1) });
    out_arcs.emplace_back(arc_collection{ construct_element(0, 1, 1), construct_element(0, 4, 1) });

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 25);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 25);

    const auto my_rank = mpi_rank_type{ 0 };

    const auto check_in_arcs = [&graph, &in_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id), in_arcs[node_id].size()) << ' ' << node_id;
        ASSERT_EQ(graph.get_in_arcs(my_rank, node_id).size(), in_arcs[node_id].size()) << ' ' << node_id;

        auto graph_set = std::set<InArc>{};
        graph_set.insert(graph.get_in_arcs(my_rank, node_id).begin(), graph.get_in_arcs(my_rank, node_id).end());

        auto arc_set = std::set<InArc>{};
        for (const auto& [key, weight] : in_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_in_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_in_arcs(my_rank, node_id)[arc_id], graph.get_in_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }
    };

    const auto check_out_arcs = [&graph, &out_arcs](const node_id_type node_id) {
        ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id), out_arcs[node_id].size()) << ' ' << node_id;
        ASSERT_EQ(graph.get_out_arcs(my_rank, node_id).size(), out_arcs[node_id].size()) << ' ' << node_id;

        auto graph_set = std::set<OutArc>{};
        graph_set.insert(graph.get_out_arcs(my_rank, node_id).begin(), graph.get_out_arcs(my_rank, node_id).end());

        auto arc_set = std::set<OutArc>{};
        for (const auto& [key, weight] : out_arcs[node_id]) {
            const auto& [rank, node] = key;
            arc_set.emplace(rank, node, weight);
        }

        ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_out_arcs(my_rank, node_id); ++arc_id) {
            ASSERT_EQ(graph.get_out_arcs(my_rank, node_id)[arc_id], graph.get_out_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }
    };

    const auto in_arc_info_distribution = graph.get_local_in_arc_info_distribution();
    const auto out_arc_info_distribution = graph.get_local_out_arc_info_distribution();

    ASSERT_EQ(in_arc_info_distribution.size(), node_id_type{ 10 });
    ASSERT_EQ(out_arc_info_distribution.size(), node_id_type{ 10 });

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        ASSERT_EQ(in_arc_info_distribution[node_id], graph.get_in_arc_info(0, node_id));
        ASSERT_EQ(out_arc_info_distribution[node_id], graph.get_out_arc_info(0, node_id));
    }

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        check_in_arcs(node_id);
        check_out_arcs(node_id);
    }

    const auto weight_in_arcs = graph.get_weight_in_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 1, 3, 2, 3, 2, 2, 5, 2, 2, 3 };
        ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_in_arcs[node_id], expected_weights[node_id]);
    }
    const auto weight_out_arcs = graph.get_weight_out_arc_distribution();
    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
        constexpr static auto expected_weights = std::array<weight_type, 10>{ 3, 2, 2, 3, 4, 3, 2, 2, 2, 2 };
        ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id), expected_weights[node_id]);
        ASSERT_EQ(weight_out_arcs[node_id], expected_weights[node_id]);
    }

    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 2 }), 4);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 3 }), 6);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 4 }), 9);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 5 }), 11);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 6 }), 13);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 7 }), 18);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 8 }), 20);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 9 }), 22);

    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 1 }), 3);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 2 }), 5);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 3 }), 7);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 4 }), 10);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 5 }), 14);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 6 }), 17);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 7 }), 19);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 8 }), 21);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 9 }), 23);

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 0 }), Vec3d(0.1, 9.8, 4.7));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 1 }), Vec3d(9.6, 5.7, 1.4));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 2 }), Vec3d(0.3, 3.9, 8.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 3 }), Vec3d(7.9, 2.8, 9.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 4 }), Vec3d(2.2, 4.9, 0.0));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 5 }), Vec3d(0.9, 5.2, 8.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 6 }), Vec3d(0.6, 5.8, 5.2));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 7 }), Vec3d(8.2, 2.0, 8.8));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 8 }), Vec3d(8.2, 4.8, 6.6));
    ASSERT_EQ(graph.get_node_position(my_rank, node_id_type{ 9 }), Vec3d(7.4, 4.4, 3.4));

    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 1 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 4 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 5 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 6 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 7 }), 1);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 8 }), 0);
    ASSERT_EQ(graph.get_node_area_localID(my_rank, node_id_type{ 9 }), 2);

    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 1 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 2 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 3 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 4 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 5 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 6 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 7 }), 0);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 8 }), 1);
    ASSERT_EQ(graph.get_node_signal_localID(my_rank, node_id_type{ 9 }), 0);
}

TEST_F(DistributedGraphTest, testLoadNodesFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto get_positions = [](const mpi_rank_type mpi_rank) {
        auto positions_rank_0 = std::vector<Vec3d>{};
        positions_rank_0.emplace_back(0.1, 9.8, 4.7);
        positions_rank_0.emplace_back(9.6, 5.7, 1.4);
        positions_rank_0.emplace_back(0.3, 3.9, 8.0);
        positions_rank_0.emplace_back(7.9, 2.8, 9.6);
        positions_rank_0.emplace_back(2.2, 4.9, 0.0);
        positions_rank_0.emplace_back(0.9, 5.2, 8.6);
        positions_rank_0.emplace_back(0.6, 5.8, 5.2);
        positions_rank_0.emplace_back(8.2, 2.0, 8.8);
        positions_rank_0.emplace_back(8.2, 4.8, 6.6);
        positions_rank_0.emplace_back(7.4, 4.4, 3.4);

        auto positions_rank_1 = std::vector<Vec3d>{};
        positions_rank_1.emplace_back(0.40, 0.31, 0.64);
        positions_rank_1.emplace_back(0.90, 0.83, 0.47);
        positions_rank_1.emplace_back(0.78, 0.45, 0.45);
        positions_rank_1.emplace_back(0.41, 0.22, 0.02);
        positions_rank_1.emplace_back(0.19, 0.93, 0.04);
        positions_rank_1.emplace_back(0.21, 0.28, 0.84);
        positions_rank_1.emplace_back(0.00, 0.41, 0.88);
        positions_rank_1.emplace_back(0.12, 0.55, 0.02);
        positions_rank_1.emplace_back(0.22, 0.91, 0.19);
        positions_rank_1.emplace_back(0.43, 0.99, 0.61);

        auto positions_rank_2 = std::vector<Vec3d>{};
        positions_rank_2.emplace_back(0.18, 0.26, 1.00);
        positions_rank_2.emplace_back(0.72, 0.59, 0.06);
        positions_rank_2.emplace_back(0.46, 0.67, 0.75);
        positions_rank_2.emplace_back(0.56, 0.44, 0.46);
        positions_rank_2.emplace_back(0.48, 0.12, 0.64);
        positions_rank_2.emplace_back(0.17, 0.79, 0.01);
        positions_rank_2.emplace_back(0.65, 0.17, 1.00);
        positions_rank_2.emplace_back(0.58, 0.87, 0.38);
        positions_rank_2.emplace_back(0.05, 0.57, 0.33);
        positions_rank_2.emplace_back(0.96, 0.44, 0.95);

        auto positions_rank_3 = std::vector<Vec3d>{};
        positions_rank_3.emplace_back(0.22, 0.29, 0.89);
        positions_rank_3.emplace_back(0.99, 0.51, 0.16);
        positions_rank_3.emplace_back(0.99, 0.72, 0.75);
        positions_rank_3.emplace_back(0.71, 0.01, 0.71);
        positions_rank_3.emplace_back(0.93, 0.68, 0.42);
        positions_rank_3.emplace_back(0.98, 0.47, 0.17);
        positions_rank_3.emplace_back(0.73, 0.48, 0.93);
        positions_rank_3.emplace_back(0.91, 0.35, 0.43);
        positions_rank_3.emplace_back(0.28, 0.12, 0.09);
        positions_rank_3.emplace_back(0.08, 0.67, 0.17);

        if (mpi_rank == 0) {
            return positions_rank_0;
        }

        if (mpi_rank == 1) {
            return positions_rank_1;
        }

        if (mpi_rank == 2) {
            return positions_rank_2;
        }

        return positions_rank_3;
    };

    const auto get_area_names = [](const mpi_rank_type mpi_rank) {
        auto area_names_rank_0 = std::vector<std::string>{};
        area_names_rank_0.emplace_back("area_1");
        area_names_rank_0.emplace_back("area_2");
        area_names_rank_0.emplace_back("area_3");

        auto area_names_rank_1 = std::vector<std::string>{};
        area_names_rank_1.emplace_back("area_1");
        area_names_rank_1.emplace_back("area_2");
        area_names_rank_1.emplace_back("area_4");

        auto area_names_rank_2 = std::vector<std::string>{};
        area_names_rank_2.emplace_back("area_1");
        area_names_rank_2.emplace_back("area_4");
        area_names_rank_2.emplace_back("area_5");

        auto area_names_rank_3 = std::vector<std::string>{};
        area_names_rank_3.emplace_back("area_2");
        area_names_rank_3.emplace_back("area_4");

        if (mpi_rank == 0) {
            return area_names_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_rank_2;
        }

        return area_names_rank_3;
    };

    const auto get_signal_types = [](const mpi_rank_type mpi_rank) {
        auto signal_types_rank_0 = std::vector<std::string>{};
        signal_types_rank_0.emplace_back("excitatory");
        signal_types_rank_0.emplace_back("inhibitory");

        auto signal_types_rank_1 = std::vector<std::string>{};
        signal_types_rank_1.emplace_back("inhibitory");
        signal_types_rank_1.emplace_back("excitatory");

        auto signal_types_rank_2 = std::vector<std::string>{};
        signal_types_rank_2.emplace_back("excitatory");
        signal_types_rank_2.emplace_back("inhibitory");

        auto signal_types_rank_3 = std::vector<std::string>{};
        signal_types_rank_3.emplace_back("excitatory");
        signal_types_rank_3.emplace_back("inhibitory");

        if (mpi_rank == 0) {
            return signal_types_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_rank_2;
        }

        return signal_types_rank_3;
    };

    const auto get_area_names_indices = [](const mpi_rank_type mpi_rank) {
        auto area_names_indices_rank_0 = std::vector<node_id_type>{};
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(2);

        auto area_names_indices_rank_1 = std::vector<node_id_type>{};
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(1);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(0);

        auto area_names_indices_rank_2 = std::vector<node_id_type>{};
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);

        auto area_names_indices_rank_3 = std::vector<node_id_type>{};
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);

        if (mpi_rank == 0) {
            return area_names_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_indices_rank_2;
        }

        return area_names_indices_rank_3;
    };

    const auto get_signal_types_indices = [](const mpi_rank_type mpi_rank) {
        auto signal_types_indices_rank_0 = std::vector<node_id_type>{};
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);

        auto signal_types_indices_rank_1 = std::vector<node_id_type>{};
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);

        auto signal_types_indices_rank_2 = std::vector<node_id_type>{};
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);

        auto signal_types_indices_rank_3 = std::vector<node_id_type>{};
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);

        if (mpi_rank == 0) {
            return signal_types_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_indices_rank_2;
        }

        return signal_types_indices_rank_3;
    };

    const auto get_in_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto in_arcs = LoadedArcs{};
        in_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2), ce(mpi_rank, 4, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 7, 1), ce(mpi_rank, 4, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 7, 2), ce(mpi_rank, 8, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 9, 1), ce(mpi_rank, 2, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 3, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 4, 2), ce(mpi_rank, 1, 3), ce(mpi_rank, 5, 9), ce(mpi_rank, 6, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 5, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 4, 1), ce(mpi_rank, 0, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 0, 1) });

        const auto other_rank = (mpi_rank + 3) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            in_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return in_arcs;
    };

    const auto get_out_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto out_arcs = LoadedArcs{};
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 6, 3) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 4, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 5, 2), ce(mpi_rank, 7, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 2, 2), ce(mpi_rank, 6, 2), ce(mpi_rank, 8, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 6, 9), ce(mpi_rank, 7, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 3, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 2), ce(mpi_rank, 4, 1) });

        const auto other_rank = (mpi_rank + 1) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            out_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return out_arcs;
    };

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto& positions = get_positions(my_rank);
    const auto& area_names = get_area_names(my_rank);
    const auto& signal_types = get_signal_types(my_rank);
    const auto& area_names_indices = get_area_names_indices(my_rank);
    const auto& signal_types_indices = get_signal_types_indices(my_rank);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    auto in_arcs = get_in_arcs(my_rank);
    auto out_arcs = get_out_arcs(my_rank);

    auto graph = DistributedGraph::construct_graph(loaded_nodes, in_arcs, out_arcs);

    ASSERT_EQ(graph.get_number_local_nodes(), 10);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 35);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 35);

    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        auto rank_in_arcs = get_in_arcs(rank);
        auto rank_out_arcs = get_out_arcs(rank);

        const auto check_in_arcs = [&graph, &rank_in_arcs](const mpi_rank_type mpi_rank, const node_id_type node_id) {
            ASSERT_EQ(graph.get_number_in_arcs(mpi_rank, node_id), rank_in_arcs[node_id].size()) << ' ' << node_id;
            ASSERT_EQ(graph.get_in_arcs(mpi_rank, node_id).size(), rank_in_arcs[node_id].size()) << ' ' << node_id;

            auto graph_set = std::set<InArc>{};
            graph_set.insert(graph.get_in_arcs(mpi_rank, node_id).begin(), graph.get_in_arcs(mpi_rank, node_id).end());

            auto arc_set = std::set<InArc>{};
            for (const auto& [key, weight] : rank_in_arcs[node_id]) {
                const auto& [source_rank, node] = key;
                arc_set.emplace(source_rank, node, weight);
            }

            ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

            for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_in_arcs(mpi_rank, node_id); ++arc_id) {
                ASSERT_EQ(graph.get_in_arcs(mpi_rank, node_id)[arc_id], graph.get_in_arc(mpi_rank, node_id, arc_id)) << ' ' << node_id;
            }
        };

        const auto check_out_arcs = [&graph, &rank_out_arcs](const mpi_rank_type mpi_rank, const node_id_type node_id) {
            ASSERT_EQ(graph.get_number_out_arcs(mpi_rank, node_id), rank_out_arcs[node_id].size()) << ' ' << node_id;
            ASSERT_EQ(graph.get_out_arcs(mpi_rank, node_id).size(), rank_out_arcs[node_id].size()) << ' ' << node_id;

            auto graph_set = std::set<OutArc>{};
            graph_set.insert(graph.get_out_arcs(mpi_rank, node_id).begin(), graph.get_out_arcs(mpi_rank, node_id).end());

            auto arc_set = std::set<OutArc>{};
            for (const auto& [key, weight] : rank_out_arcs[node_id]) {
                const auto& [target_rank, node] = key;
                arc_set.emplace(target_rank, node, weight);
            }

            ASSERT_EQ(graph_set, arc_set) << ' ' << node_id;

            for (auto arc_id = arc_id_type{ 0 }; arc_id < graph.get_number_out_arcs(mpi_rank, node_id); ++arc_id) {
                ASSERT_EQ(graph.get_out_arcs(mpi_rank, node_id)[arc_id], graph.get_out_arc(mpi_rank, node_id, arc_id)) << ' ' << node_id;
            }
        };

        const auto check_weight_in_arcs = [&graph](const mpi_rank_type mpi_rank, const node_id_type node_id) {
            const auto _in_arcs = graph.get_in_arcs(mpi_rank, node_id);
            const auto expected_weight = std::accumulate(_in_arcs.begin(), _in_arcs.end(), weight_type{ 0 }, [](const auto& sum, const auto& arc) {
                return sum + arc.weight;
            });

            ASSERT_EQ(expected_weight, graph.get_weight_in_arcs(mpi_rank, node_id));
        };

        const auto check_weight_out_arcs = [&graph](const mpi_rank_type mpi_rank, const node_id_type node_id) {
            const auto _out_arcs = graph.get_out_arcs(mpi_rank, node_id);
            const auto expected_weight = std::accumulate(_out_arcs.begin(), _out_arcs.end(), weight_type{ 0 }, [](const auto& sum, const auto& arc) {
                return sum + arc.weight;
            });

            ASSERT_EQ(expected_weight, graph.get_weight_out_arcs(mpi_rank, node_id));
        };

        for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; ++node_id) {
            check_in_arcs(rank, node_id);
            check_out_arcs(rank, node_id);

            check_weight_in_arcs(rank, node_id);
            check_weight_out_arcs(rank, node_id);
        }
    }

    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 0 }), 0);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 1 }), 2);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 2 }), 6);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 3 }), 9);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 4 }), 13);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 5 }), 16);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 6 }), 19);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 7 }), 25);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 8 }), 28);
        ASSERT_EQ(graph.get_prefix_in_arcs(rank, node_id_type{ 9 }), 31);

        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 0 }), 0);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 1 }), 4);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 2 }), 7);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 3 }), 10);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 4 }), 14);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 5 }), 19);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 6 }), 23);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 7 }), 26);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 8 }), 29);
        ASSERT_EQ(graph.get_prefix_out_arcs(rank, node_id_type{ 9 }), 32);
    }

    ASSERT_TRUE(graph.get_local_area_names() == std::span<const std::string>{ area_names });
    ASSERT_TRUE(graph.get_local_signal_types() == std::span<const std::string>{ signal_types });

    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        const auto& rank_positions = get_positions(rank);

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            ASSERT_EQ(graph.get_node_position(rank, node_id), rank_positions[node_id]);
        }
    }

    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        const auto& rank_area_names_indices = get_area_names_indices(rank);

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            ASSERT_EQ(graph.get_node_area_localID(rank, node_id), rank_area_names_indices[node_id]);
        }
    }

    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        const auto& rank_signal_types_indices = get_signal_types_indices(rank);

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            ASSERT_EQ(graph.get_node_signal_localID(rank, node_id), rank_signal_types_indices[node_id]);
        }
    }
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcsFileUndirected) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
    auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
    auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };

    // Three nodes forming a directed cycle 0 -> 1 -> 2 -> 0.
    nodes_file << "1 0.1 0.2 0.3 area_1 excitatory\n";
    nodes_file << "2 0.4 0.5 0.6 area_1 excitatory\n";
    nodes_file << "3 0.7 0.8 0.9 area_1 excitatory\n";

    // in arcs: target_rank target_id source_rank source_id weight (ids are 1-based)
    in_arcs_file << "0 1 0 3 1\n"; // 0 <- 2
    in_arcs_file << "0 2 0 1 1\n"; // 1 <- 0
    in_arcs_file << "0 3 0 2 1\n"; // 2 <- 1

    // out arcs: target_rank target_id source_rank source_id weight (ids are 1-based)
    out_arcs_file << "0 2 0 1 1\n"; // 0 -> 1
    out_arcs_file << "0 3 0 2 1\n"; // 1 -> 2
    out_arcs_file << "0 1 0 3 1\n"; // 2 -> 0

    nodes_file.flush();
    nodes_file.close();

    in_arcs_file.flush();
    in_arcs_file.close();

    out_arcs_file.flush();
    out_arcs_file.close();

    // undirected = true: the in and out neighbours of each node are disjoint here, so synchronize_arcs produces
    // more arcs (6) than either input set (3). Before the window-sizing fix this overflowed the RMA windows.
    auto graph = DistributedGraph::construct_graph("./input/", false, true, false, "");

    ASSERT_EQ(graph.get_number_local_nodes(), 3);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 6);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 6);

    const auto my_rank = mpi_rank_type{ 0 };

    // undirected neighbours of each node: union of its out- and in-neighbours
    const auto expected_neighbours = std::array<std::set<node_id_type>, 3>{
        std::set<node_id_type>{ 1, 2 }, // node 0: out 1, in 2
        std::set<node_id_type>{ 0, 2 }, // node 1: out 2, in 0
        std::set<node_id_type>{ 0, 1 }, // node 2: out 0, in 1
    };

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 3 }; ++node_id) {
        const auto in_span = graph.get_in_arcs(my_rank, node_id);
        const auto out_span = graph.get_out_arcs(my_rank, node_id);

        ASSERT_EQ(in_span.size(), expected_neighbours[node_id].size()) << ' ' << node_id;
        ASSERT_EQ(out_span.size(), expected_neighbours[node_id].size()) << ' ' << node_id;

        auto in_neighbours = std::set<node_id_type>{};
        auto in_weight = weight_type{ 0 };
        for (const auto& arc : in_span) {
            ASSERT_EQ(arc.source_rank, my_rank) << ' ' << node_id;
            in_neighbours.insert(arc.source_id);
            in_weight += arc.weight;
        }

        auto out_neighbours = std::set<node_id_type>{};
        auto out_weight = weight_type{ 0 };
        for (const auto& arc : out_span) {
            ASSERT_EQ(arc.target_rank, my_rank) << ' ' << node_id;
            out_neighbours.insert(arc.target_id);
            out_weight += arc.weight;
        }

        ASSERT_EQ(in_neighbours, expected_neighbours[node_id]) << ' ' << node_id;
        // undirected: incoming and outgoing arcs are identical
        ASSERT_EQ(out_neighbours, expected_neighbours[node_id]) << ' ' << node_id;

        ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id), in_weight) << ' ' << node_id;
        ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id), out_weight) << ' ' << node_id;
    }

    // With two arcs per node the prefixes must be the running arc count.
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 1 }), 2);
    ASSERT_EQ(graph.get_prefix_in_arcs(my_rank, node_id_type{ 2 }), 4);

    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 0 }), 0);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 1 }), 2);
    ASSERT_EQ(graph.get_prefix_out_arcs(my_rank, node_id_type{ 2 }), 4);
}

TEST_F(DistributedGraphTest, testLoadArcsRejectsOutOfRangeEndpoints) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
    auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
    auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };

    nodes_file << "1 0.1 0.2 0.3 area_1 excitatory\n";
    nodes_file << "2 0.4 0.5 0.6 area_1 excitatory\n";
    nodes_file << "3 0.7 0.8 0.9 area_1 excitatory\n";

    // One valid in arc and endpoints that must be rejected against the node distribution [3].
    in_arcs_file << "0 1 0 2 1\n";  // valid: node 0 <- node 1
    in_arcs_file << "0 1 0 99 1\n"; // invalid: source id out of range
    in_arcs_file << "0 1 5 1 1\n";  // invalid: source rank out of range

    // One valid out arc and endpoints that must be rejected.
    out_arcs_file << "0 2 0 1 1\n";  // valid: node 0 -> node 1
    out_arcs_file << "0 99 0 1 1\n"; // invalid: target id out of range
    out_arcs_file << "5 1 0 1 1\n";  // invalid: target rank out of range

    nodes_file.flush();
    nodes_file.close();

    in_arcs_file.flush();
    in_arcs_file.close();

    out_arcs_file.flush();
    out_arcs_file.close();

    auto graph = DistributedGraph::construct_graph("./input/", false, false, false, "");

    ASSERT_EQ(graph.get_number_local_nodes(), 3);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 1);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 1);

    const auto my_rank = mpi_rank_type{ 0 };

    const auto in_span = graph.get_in_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(in_span.size(), 1);
    ASSERT_EQ(in_span[0].source_rank, my_rank);
    ASSERT_EQ(in_span[0].source_id, node_id_type{ 1 });

    const auto out_span = graph.get_out_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(out_span.size(), 1);
    ASSERT_EQ(out_span[0].target_rank, my_rank);
    ASSERT_EQ(out_span[0].target_id, node_id_type{ 1 });
}

TEST_F(DistributedGraphTest, testUploadArcsSkipsZeroWeightArcs) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.1, 0.2, 0.3);
    positions.emplace_back(0.4, 0.5, 0.6);
    positions.emplace_back(0.7, 0.8, 0.9);

    auto area_names = std::vector<std::string>{ "area_1" };
    auto signal_types = std::vector<std::string>{ "excitatory" };
    auto area_names_indices = std::vector<node_id_type>{ 0, 0, 0 };
    auto signal_types_indices = std::vector<node_id_type>{ 0, 0, 0 };

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;
    constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    // node 0 has one real in arc and one zero-weight in arc; the zero-weight arc must be skipped on upload
    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ ce(0, 1, 5), ce(0, 2, 0) });
    in_arcs.emplace_back(arc_collection{});
    in_arcs.emplace_back(arc_collection{});

    // node 0 has one zero-weight out arc (skipped) and one real out arc
    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ ce(0, 1, 0), ce(0, 2, 7) });
    out_arcs.emplace_back(arc_collection{});
    out_arcs.emplace_back(arc_collection{});

    auto graph = DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);

    const auto my_rank = mpi_rank_type{ 0 };

    // the zero-weight arcs are not stored, so the counts are one below the number of map entries
    ASSERT_EQ(graph.get_number_local_in_arcs(), 1);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 1);

    ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id_type{ 0 }), 1);
    const auto in_span = graph.get_in_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(in_span.size(), 1);
    ASSERT_EQ(in_span[0].source_id, node_id_type{ 1 });
    ASSERT_EQ(in_span[0].weight, weight_type{ 5 });
    ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id_type{ 0 }), weight_type{ 5 });

    ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id_type{ 0 }), 1);
    const auto out_span = graph.get_out_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(out_span.size(), 1);
    ASSERT_EQ(out_span[0].target_id, node_id_type{ 2 });
    ASSERT_EQ(out_span[0].weight, weight_type{ 7 });
    ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id_type{ 0 }), weight_type{ 7 });
}

TEST_F(DistributedGraphTest, testLockAndUnlockAllRmaWindows) {
    // This test is collective: it runs on every rank so that the barriers inside lock/unlock match.
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.1, 0.2, 0.3);
    positions.emplace_back(0.4, 0.5, 0.6);

    auto area_names = std::vector<std::string>{ "area_1" };
    auto signal_types = std::vector<std::string>{ "excitatory" };
    auto area_names_indices = std::vector<node_id_type>{ 0, 0 };
    auto signal_types_indices = std::vector<node_id_type>{ 0, 0 };

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;
    const auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ ce(my_rank, 1, 1) });
    in_arcs.emplace_back(arc_collection{});

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ ce(my_rank, 1, 1) });
    out_arcs.emplace_back(arc_collection{});

    auto graph = DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);

    // Locking and unlocking all windows must succeed and leave the graph usable afterwards.
    graph.lock_all_rma_windows();
    ASSERT_EQ(graph.get_number_local_nodes(), 2);
    graph.unlock_all_rma_windows();

    ASSERT_EQ(graph.get_number_local_nodes(), 2);
    ASSERT_EQ(graph.get_number_in_arcs(my_rank, node_id_type{ 0 }), 1);
    ASSERT_EQ(graph.get_number_out_arcs(my_rank, node_id_type{ 0 }), 1);
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcsFileRemoveSelfArcs) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    {
        auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
        nodes_file << "1 0.1 0.2 0.3 area_1 excitatory\n";
        nodes_file << "2 0.4 0.5 0.6 area_1 excitatory\n";
        nodes_file << "3 0.7 0.8 0.9 area_1 excitatory\n";
    }
    {
        auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
        in_arcs_file << "0 1 0 1 5\n"; // node 0 <- node 0 : self arc (must be removed)
        in_arcs_file << "0 1 0 2 3\n"; // node 0 <- node 1 : kept
    }
    {
        auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };
        out_arcs_file << "0 1 0 1 5\n"; // node 0 -> node 0 : self arc (must be removed)
        out_arcs_file << "0 2 0 1 7\n"; // node 0 -> node 1 : kept
    }

    // remove_self_arcs = true, undirected = false, one_weight = false
    auto graph = DistributedGraph::construct_graph("./input/", true, false, false, "");

    ASSERT_EQ(graph.get_number_local_nodes(), 3);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 1);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 1);

    const auto my_rank = mpi_rank_type{ 0 };

    const auto in_span = graph.get_in_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(in_span.size(), 1);
    ASSERT_EQ(in_span[0].source_rank, my_rank);
    ASSERT_EQ(in_span[0].source_id, node_id_type{ 1 });
    ASSERT_EQ(in_span[0].weight, weight_type{ 3 });

    const auto out_span = graph.get_out_arcs(my_rank, node_id_type{ 0 });
    ASSERT_EQ(out_span.size(), 1);
    ASSERT_EQ(out_span[0].target_rank, my_rank);
    ASSERT_EQ(out_span[0].target_id, node_id_type{ 1 });
    ASSERT_EQ(out_span[0].weight, weight_type{ 7 });
}

TEST_F(DistributedGraphTest, testLoadNodesAndArcsFileUndirectedWeightsOne) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./input/positions");
    std::filesystem::create_directories("./input/network");

    {
        auto nodes_file = std::ofstream{ "./input/positions/rank_0_positions.txt" };
        nodes_file << "1 0.1 0.2 0.3 area_1 excitatory\n";
        nodes_file << "2 0.4 0.5 0.6 area_1 excitatory\n";
        nodes_file << "3 0.7 0.8 0.9 area_1 excitatory\n";
    }
    {
        // in arcs of a directed cycle 0 -> 1 -> 2 -> 0 with varying weights
        auto in_arcs_file = std::ofstream{ "./input/network/rank_0_in_network.txt" };
        in_arcs_file << "0 1 0 3 2\n"; // 0 <- 2
        in_arcs_file << "0 2 0 1 3\n"; // 1 <- 0
        in_arcs_file << "0 3 0 2 4\n"; // 2 <- 1
    }
    {
        auto out_arcs_file = std::ofstream{ "./input/network/rank_0_out_network.txt" };
        out_arcs_file << "0 2 0 1 5\n"; // 0 -> 1
        out_arcs_file << "0 3 0 2 6\n"; // 1 -> 2
        out_arcs_file << "0 1 0 3 7\n"; // 2 -> 0
    }

    // undirected = true and one_weight = true: arcs are symmetrized and every weight is set to one.
    auto graph = DistributedGraph::construct_graph("./input/", false, true, true, "");

    ASSERT_EQ(graph.get_number_local_nodes(), 3);
    ASSERT_EQ(graph.get_number_local_in_arcs(), 6);
    ASSERT_EQ(graph.get_number_local_out_arcs(), 6);

    const auto my_rank = mpi_rank_type{ 0 };

    // undirected neighbours of each node: union of its out- and in-neighbours
    const auto expected_neighbours = std::array<std::set<node_id_type>, 3>{
        std::set<node_id_type>{ 1, 2 }, // node 0: out 1, in 2
        std::set<node_id_type>{ 0, 2 }, // node 1: out 2, in 0
        std::set<node_id_type>{ 0, 1 }, // node 2: out 0, in 1
    };

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 3 }; ++node_id) {
        const auto in_span = graph.get_in_arcs(my_rank, node_id);
        const auto out_span = graph.get_out_arcs(my_rank, node_id);

        ASSERT_EQ(in_span.size(), 2) << ' ' << node_id;
        ASSERT_EQ(out_span.size(), 2) << ' ' << node_id;

        auto in_neighbours = std::set<node_id_type>{};
        for (const auto& arc : in_span) {
            ASSERT_EQ(arc.weight, weight_type{ 1 }) << ' ' << node_id;
            in_neighbours.insert(arc.source_id);
        }

        auto out_neighbours = std::set<node_id_type>{};
        for (const auto& arc : out_span) {
            ASSERT_EQ(arc.weight, weight_type{ 1 }) << ' ' << node_id;
            out_neighbours.insert(arc.target_id);
        }

        ASSERT_EQ(in_neighbours, expected_neighbours[node_id]) << ' ' << node_id;
        ASSERT_EQ(out_neighbours, expected_neighbours[node_id]) << ' ' << node_id;

        // two arcs of weight one each
        ASSERT_EQ(graph.get_weight_in_arcs(my_rank, node_id), weight_type{ 2 }) << ' ' << node_id;
        ASSERT_EQ(graph.get_weight_out_arcs(my_rank, node_id), weight_type{ 2 }) << ' ' << node_id;
    }
}

TEST_F(DistributedGraphTest, testArcInfoLayout) {
    // The whole point of the struct is that a node's arc count and arc prefix fit into eight bytes,
    // i.e., that both can be fetched with a single remote memory access.
    ASSERT_EQ(sizeof(ArcInfo), 8U);
    ASSERT_EQ(alignof(ArcInfo), alignof(arc_id_type));

    constexpr auto default_arc_info = ArcInfo{};
    ASSERT_EQ(default_arc_info.number_arcs, arc_id_type{ 0 });
    ASSERT_EQ(default_arc_info.prefix_arcs, arc_id_type{ 0 });

    constexpr auto arc_info = ArcInfo{ 3, 7 };
    ASSERT_EQ(arc_info.number_arcs, arc_id_type{ 3 });
    ASSERT_EQ(arc_info.prefix_arcs, arc_id_type{ 7 });

    ASSERT_EQ(arc_info, (ArcInfo{ 3, 7 }));
    ASSERT_NE(arc_info, (ArcInfo{ 7, 3 }));
    ASSERT_NE(arc_info, default_arc_info);

    auto stream = std::ostringstream{};
    stream << arc_info;
    ASSERT_EQ(stream.str(), "(3, 7)");
}

TEST_F(DistributedGraphTest, testArcInfo) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    constexpr auto number_nodes = node_id_type{ 10 };

    // The arcs are stored in ascending node order, so the prefix of a node is the number of arcs of all previous nodes.
    constexpr auto expected_in_arc_infos = std::array<ArcInfo, number_nodes>{ ArcInfo{ 1, 0 },  ArcInfo{ 3, 1 },  ArcInfo{ 2, 4 },  ArcInfo{ 3, 6 },
                                                                              ArcInfo{ 2, 9 },  ArcInfo{ 2, 11 }, ArcInfo{ 5, 13 }, ArcInfo{ 2, 18 },
                                                                              ArcInfo{ 2, 20 }, ArcInfo{ 3, 22 } };

    constexpr auto expected_out_arc_infos = std::array<ArcInfo, number_nodes>{ ArcInfo{ 3, 0 },  ArcInfo{ 2, 3 },  ArcInfo{ 2, 5 },  ArcInfo{ 3, 7 },
                                                                               ArcInfo{ 4, 10 }, ArcInfo{ 3, 14 }, ArcInfo{ 2, 17 }, ArcInfo{ 2, 19 },
                                                                               ArcInfo{ 2, 21 }, ArcInfo{ 2, 23 } };

    const auto my_rank = mpi_rank_type{ 0 };

    const auto in_arc_info_distribution = graph.get_local_in_arc_info_distribution();
    const auto out_arc_info_distribution = graph.get_local_out_arc_info_distribution();

    ASSERT_EQ(in_arc_info_distribution.size(), number_nodes);
    ASSERT_EQ(out_arc_info_distribution.size(), number_nodes);

    for (auto node_id = node_id_type{ 0 }; node_id < number_nodes; ++node_id) {
        const auto in_arc_info = graph.get_in_arc_info(my_rank, node_id);

        ASSERT_EQ(in_arc_info, expected_in_arc_infos[node_id]) << ' ' << node_id;
        ASSERT_EQ(in_arc_info, in_arc_info_distribution[node_id]) << ' ' << node_id;

        // The single accessors must return the respective member of the packed info.
        ASSERT_EQ(in_arc_info.number_arcs, graph.get_number_in_arcs(my_rank, node_id)) << ' ' << node_id;
        ASSERT_EQ(in_arc_info.prefix_arcs, graph.get_prefix_in_arcs(my_rank, node_id)) << ' ' << node_id;

        // The arcs of the node must be stored in [prefix_arcs, prefix_arcs + number_arcs) of the arc window.
        const auto in_span = graph.get_in_arcs(my_rank, node_id);
        ASSERT_EQ(in_span.size(), in_arc_info.number_arcs) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < in_arc_info.number_arcs; ++arc_id) {
            ASSERT_EQ(in_span[arc_id], graph.get_in_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }

        const auto out_arc_info = graph.get_out_arc_info(my_rank, node_id);

        ASSERT_EQ(out_arc_info, expected_out_arc_infos[node_id]) << ' ' << node_id;
        ASSERT_EQ(out_arc_info, out_arc_info_distribution[node_id]) << ' ' << node_id;

        ASSERT_EQ(out_arc_info.number_arcs, graph.get_number_out_arcs(my_rank, node_id)) << ' ' << node_id;
        ASSERT_EQ(out_arc_info.prefix_arcs, graph.get_prefix_out_arcs(my_rank, node_id)) << ' ' << node_id;

        const auto out_span = graph.get_out_arcs(my_rank, node_id);
        ASSERT_EQ(out_span.size(), out_arc_info.number_arcs) << ' ' << node_id;

        for (auto arc_id = arc_id_type{ 0 }; arc_id < out_arc_info.number_arcs; ++arc_id) {
            ASSERT_EQ(out_span[arc_id], graph.get_out_arc(my_rank, node_id, arc_id)) << ' ' << node_id;
        }
    }

    // The infos of the last node must cover the arc windows completely.
    const auto& last_in_arc_info = in_arc_info_distribution[number_nodes - 1];
    ASSERT_EQ(last_in_arc_info.prefix_arcs + last_in_arc_info.number_arcs, graph.get_number_local_in_arcs());

    const auto& last_out_arc_info = out_arc_info_distribution[number_nodes - 1];
    ASSERT_EQ(last_out_arc_info.prefix_arcs + last_out_arc_info.number_arcs, graph.get_number_local_out_arcs());
}

TEST_F(DistributedGraphTest, testArcInfoFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    constexpr auto number_nodes = node_id_type{ 10 };

    // Every rank stores the same arc structure, so the same infos are expected for all four ranks.
    constexpr auto expected_in_arc_infos = std::array<ArcInfo, number_nodes>{ ArcInfo{ 2, 0 },  ArcInfo{ 4, 2 },  ArcInfo{ 3, 6 },  ArcInfo{ 4, 9 },
                                                                              ArcInfo{ 3, 13 }, ArcInfo{ 3, 16 }, ArcInfo{ 6, 19 }, ArcInfo{ 3, 25 },
                                                                              ArcInfo{ 3, 28 }, ArcInfo{ 4, 31 } };

    constexpr auto expected_out_arc_infos = std::array<ArcInfo, number_nodes>{ ArcInfo{ 4, 0 },  ArcInfo{ 3, 4 },  ArcInfo{ 3, 7 },  ArcInfo{ 4, 10 },
                                                                               ArcInfo{ 5, 14 }, ArcInfo{ 4, 19 }, ArcInfo{ 3, 23 }, ArcInfo{ 3, 26 },
                                                                               ArcInfo{ 3, 29 }, ArcInfo{ 3, 32 } };

    // Reading the info of a node of another rank must yield the same values as reading it locally.
    for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
        for (auto node_id = node_id_type{ 0 }; node_id < number_nodes; ++node_id) {
            const auto in_arc_info = graph.get_in_arc_info(rank, node_id);

            ASSERT_EQ(in_arc_info, expected_in_arc_infos[node_id]) << rank << ' ' << node_id;
            ASSERT_EQ(in_arc_info.number_arcs, graph.get_number_in_arcs(rank, node_id)) << rank << ' ' << node_id;
            ASSERT_EQ(in_arc_info.prefix_arcs, graph.get_prefix_in_arcs(rank, node_id)) << rank << ' ' << node_id;

            const auto in_span = graph.get_in_arcs(rank, node_id);
            ASSERT_EQ(in_span.size(), in_arc_info.number_arcs) << rank << ' ' << node_id;

            for (auto arc_id = arc_id_type{ 0 }; arc_id < in_arc_info.number_arcs; ++arc_id) {
                ASSERT_EQ(in_span[arc_id], graph.get_in_arc(rank, node_id, arc_id)) << rank << ' ' << node_id;
            }

            const auto out_arc_info = graph.get_out_arc_info(rank, node_id);

            ASSERT_EQ(out_arc_info, expected_out_arc_infos[node_id]) << rank << ' ' << node_id;
            ASSERT_EQ(out_arc_info.number_arcs, graph.get_number_out_arcs(rank, node_id)) << rank << ' ' << node_id;
            ASSERT_EQ(out_arc_info.prefix_arcs, graph.get_prefix_out_arcs(rank, node_id)) << rank << ' ' << node_id;

            const auto out_span = graph.get_out_arcs(rank, node_id);
            ASSERT_EQ(out_span.size(), out_arc_info.number_arcs) << rank << ' ' << node_id;

            for (auto arc_id = arc_id_type{ 0 }; arc_id < out_arc_info.number_arcs; ++arc_id) {
                ASSERT_EQ(out_span[arc_id], graph.get_out_arc(rank, node_id, arc_id)) << rank << ' ' << node_id;
            }
        }
    }

    // The local windows must contain what the other ranks read from them.
    const auto in_arc_info_distribution = graph.get_local_in_arc_info_distribution();
    const auto out_arc_info_distribution = graph.get_local_out_arc_info_distribution();

    ASSERT_EQ(in_arc_info_distribution.size(), number_nodes);
    ASSERT_EQ(out_arc_info_distribution.size(), number_nodes);

    for (auto node_id = node_id_type{ 0 }; node_id < number_nodes; ++node_id) {
        ASSERT_EQ(in_arc_info_distribution[node_id], expected_in_arc_infos[node_id]) << ' ' << node_id;
        ASSERT_EQ(out_arc_info_distribution[node_id], expected_out_arc_infos[node_id]) << ' ' << node_id;
    }
}
