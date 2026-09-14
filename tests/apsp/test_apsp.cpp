/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_apsp.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "graph/GraphTypes.h"
#include "metrics/paths/AllPairsShortestPath.h"
#include "utility/Vec3.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
// 4 nodes on one rank: 0 -> 1 -> 2 -> 0 form a directed cycle (one strongly connected component),
// node 3 is isolated. Unlike the shared fixtures (all fully connected) this produces disconnected
// pairs and two distinct cluster sizes.
[[nodiscard]] DistributedGraph make_disconnected_one_rank_graph() {
    auto positions = std::vector<Vec3d>{};
    positions.emplace_back(0.0, 0.0, 0.0);
    positions.emplace_back(1.0, 0.0, 0.0);
    positions.emplace_back(2.0, 0.0, 0.0);
    positions.emplace_back(3.0, 0.0, 0.0);

    auto area_names = std::vector<std::string>{ "area_1" };
    auto signal_types = std::vector<std::string>{ "excitatory" };

    auto area_names_indices = std::vector<node_id_type>{ 0, 0, 0, 0 };
    auto signal_types_indices = std::vector<node_id_type>{ 0, 0, 0, 0 };

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    using arc_collection = LoadedArcs::value_type;
    constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    auto in_arcs = LoadedArcs{};
    in_arcs.emplace_back(arc_collection{ ce(0, 2, 1) }); // 2 -> 0
    in_arcs.emplace_back(arc_collection{ ce(0, 0, 1) }); // 0 -> 1
    in_arcs.emplace_back(arc_collection{ ce(0, 1, 1) }); // 1 -> 2
    in_arcs.emplace_back(arc_collection{});              // node 3 is isolated

    auto out_arcs = LoadedArcs{};
    out_arcs.emplace_back(arc_collection{ ce(0, 1, 1) }); // 0 -> 1
    out_arcs.emplace_back(arc_collection{ ce(0, 2, 1) }); // 1 -> 2
    out_arcs.emplace_back(arc_collection{ ce(0, 0, 1) }); // 2 -> 0
    out_arcs.emplace_back(arc_collection{});              // node 3 is isolated

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}
} // namespace

TEST_F(APSPTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 3.1);
    ASSERT_NEAR(average_efficiency, 0.4222486772486772, 1e-6);
    ASSERT_EQ(diameter, 7);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testDisconnected) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = make_disconnected_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // {0,1,2} is a directed 3-cycle, node 3 is isolated:
    //   from each of 0,1,2: distances 1 and 2 to the other two -> sum 3, efficiency 1/1 + 1/2 = 1.5, eccentricity 2, node 3 unreachable
    //   from 3: nothing reachable -> nodes 0,1,2 unreachable
    // reached ordered pairs = 3 * 2 = 6; total distance = 9; total efficiency = 4.5; unreachable pairs = 4*3 - 6 = 6
    ASSERT_EQ(average_shortest_path_len, 1.5);   // 9 / 6
    ASSERT_NEAR(average_efficiency, 0.75, 1e-6); // 4.5 / 6
    ASSERT_EQ(diameter, 2);
    ASSERT_EQ(number_disconnected_pairs, 6);
    ASSERT_EQ(cluster_sizes.size(), 2);
    ASSERT_TRUE(cluster_sizes.contains(3)); // nodes 0,1,2 each reach 3 nodes (including themselves)
    ASSERT_EQ(cluster_sizes.at(3), 3);
    ASSERT_TRUE(cluster_sizes.contains(1)); // node 3 reaches only itself
    ASSERT_EQ(cluster_sizes.at(1), 1);
}

TEST_F(APSPTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.6, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.7222222222222222, 1e-6);
    ASSERT_EQ(diameter, 3);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_EQ(average_efficiency, 1.0);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 19.784615384615385);
    ASSERT_NEAR(average_efficiency, 0.13387777556694200, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 19.784615384615385);
    ASSERT_NEAR(average_efficiency, 0.13387777556694200, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPInverseTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.7043209876543211, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.89050315072273489, 1e-6);
    ASSERT_EQ(diameter, 4.5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.6, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.7222222222222222, 1e-6);
    ASSERT_EQ(diameter, 3);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_EQ(average_efficiency, 1.0);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 1.7130795130795136, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.2097533633823983, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPInverseTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 1.7130795130795136, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.2097533633823983, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

// The seven-rank graphs are the first fixtures with disconnected pairs: their ranks 2 and 5 form a
// component of their own and rank 3 holds isolated nodes, sinks and sources. Every source therefore
// reaches only its own part of the graph, which is what the cluster sizes record.

TEST_F(APSPTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The 40 nodes of the large component reach 40 nodes each, the 20 of the small one reach 20, the
    // six isolated nodes and sinks reach only themselves, and the four sources reach their target as
    // well: 70 * 70 - 2014 = 2886 ordered pairs stay unreachable.
    // The values serve as standard for the seven-rank test
    ASSERT_EQ(average_shortest_path_len, 17.587448559670783);
    ASSERT_NEAR(average_efficiency, 0.15517143563687033, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 2886);
    ASSERT_EQ(cluster_sizes.size(), 4);
    ASSERT_EQ(cluster_sizes.at(1), 6);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // Symmetrizing turns the sinks and sources of rank 3 into the components {3, 6, 9}, {4, 7} and
    // {5, 8}, so three nodes reach three, four nodes reach two, and only the three isolated ones
    // stay alone.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.4061538461538463, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.50605128205128700, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 2880);
    ASSERT_EQ(cluster_sizes.size(), 5);
    ASSERT_EQ(cluster_sizes.at(1), 3);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(3), 3);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The graph is complete and every arc has weight one, so every one of the 63 * 62 ordered pairs
    // is one step apart.
    // The values serve as standard for the seven-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_NEAR(average_efficiency, 1.0, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_EQ(cluster_sizes.at(63), 63);
}

TEST_F(APSPTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 17.587448559670783);
    ASSERT_NEAR(average_efficiency, 0.15517143563687033, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 2886);
    ASSERT_EQ(cluster_sizes.size(), 4);
    ASSERT_EQ(cluster_sizes.at(1), 6);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.4061538461538463, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.50605128205128700, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 2880);
    ASSERT_EQ(cluster_sizes.size(), 5);
    ASSERT_EQ(cluster_sizes.at(1), 3);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(3), 3);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_NEAR(average_efficiency, 1.0, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_EQ(cluster_sizes.at(63), 63);
}

TEST_F(APSPInverseTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The inverse metric makes the heavy ring arcs the cheapest ones, so the distances shrink; which
    // pairs are reachable does not change.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_shortest_path_len, 1.7007328947915260, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.2430679013912713, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 2886);
    ASSERT_EQ(cluster_sizes.size(), 4);
    ASSERT_EQ(cluster_sizes.at(1), 6);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // Every weight is one, so the inverse metric agrees with the regular one.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.4061538461538463, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.50605128205128700, 1e-6);
    ASSERT_NEAR(diameter, 5.0, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 2880);
    ASSERT_EQ(cluster_sizes.size(), 5);
    ASSERT_EQ(cluster_sizes.at(1), 3);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(3), 3);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // Every arc has weight one, so inverting the weights changes nothing.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_shortest_path_len, 1.0, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.0, 1e-6);
    ASSERT_NEAR(diameter, 1.0, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_EQ(cluster_sizes.at(63), 63);
}

TEST_F(APSPInverseTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 1.7007328947915260, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.2430679013912713, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 2886);
    ASSERT_EQ(cluster_sizes.size(), 4);
    ASSERT_EQ(cluster_sizes.at(1), 6);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.4061538461538463, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.50605128205128700, 1e-6);
    ASSERT_NEAR(diameter, 5.0, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 2880);
    ASSERT_EQ(cluster_sizes.size(), 5);
    ASSERT_EQ(cluster_sizes.at(1), 3);
    ASSERT_EQ(cluster_sizes.at(2), 4);
    ASSERT_EQ(cluster_sizes.at(3), 3);
    ASSERT_EQ(cluster_sizes.at(20), 20);
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 1.0, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.0, 1e-6);
    ASSERT_NEAR(diameter, 1.0, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_EQ(cluster_sizes.at(63), 63);
}
