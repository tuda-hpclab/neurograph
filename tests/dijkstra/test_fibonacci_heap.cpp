/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fibonacci_heap.h"

#include "Types.h"

#include "metrics/paths/AllPairsShortestPath.h"

#include <mpi-wrapper/core/MPIInfo.h>

namespace {
constexpr auto queue_type = DijkstraQueueType::FibonacciHeap;
} // namespace

TEST_F(DijkstraFibonacciHeapTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 3.1);
    ASSERT_NEAR(average_efficiency, 0.4222486772486772, 1e-6);
    ASSERT_EQ(diameter, 7);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(DijkstraFibonacciHeapTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.6, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.7222222222222222, 1e-6);
    ASSERT_EQ(diameter, 3);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(DijkstraFibonacciHeapTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_EQ(average_efficiency, 1.0);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(DijkstraFibonacciHeapTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 19.784615384615385);
    ASSERT_NEAR(average_efficiency, 0.13387777556694200, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(DijkstraFibonacciHeapTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testStandardInverse) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph, queue_type);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.7043209876543211, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.89050315072273489, 1e-6);
    ASSERT_EQ(diameter, 4.5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(DijkstraFibonacciHeapTest, testStandardFourRanksInverse) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The graph falls into three parts, so this is the first fixture with unreachable pairs.
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

TEST_F(DijkstraFibonacciHeapTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

    // The values serve as standard for the seven-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_NEAR(average_efficiency, 1.0, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_EQ(cluster_sizes.at(63), 63);
}

TEST_F(DijkstraFibonacciHeapTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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

TEST_F(DijkstraFibonacciHeapTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph, queue_type);

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
