/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_all_pairs_distances.h"

#include "metrics/geometry/AllPairsDistances.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(AllPairsDistancesTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    // A graph with fewer than two nodes has no node pairs; the functions must return 0.0 instead of dividing by zero.
    ASSERT_DOUBLE_EQ(AllPairsDistances::compute_average_pair_distance(graph), 0.0);
    ASSERT_DOUBLE_EQ(AllPairsDistances::all_compute_average_pair_distance(graph), 0.0);
}

TEST_F(AllPairsDistancesTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.396961, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.396961, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.056421, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

namespace {
/**
 * @brief Checks all six pairwise-distance entry points against the expected values. The plain
 *		variants return their result only on the root rank, the collective ones on every rank.
 * @param graph The graph
 * @param expected_average The expected average distance between two distinct nodes
 * @param expected_minimum The expected smallest distance
 * @param expected_maximum The expected largest distance
 */
void expect_pair_distances(const DistributedGraph& graph, const double expected_average, const double expected_minimum,
                           const double expected_maximum) {
    // All six are collectives, so they run before the first assertion: a failing assertion returns
    // early, and a rank that skipped a collective would leave the others waiting forever
    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_NEAR(average_distance, expected_average, 1e-6);
        ASSERT_NEAR(minimum_distance, expected_minimum, 1e-6);
        ASSERT_NEAR(maximum_distance, expected_maximum, 1e-6);
    }

    ASSERT_NEAR(all_average_distance, expected_average, 1e-6);
    ASSERT_NEAR(all_minimum_distance, expected_minimum, 1e-6);
    ASSERT_NEAR(all_maximum_distance, expected_maximum, 1e-6);
}
} // namespace

// The standard and the undirected, unit-weight seven-rank graph hold exactly the same nodes at the
// same positions, so their pairwise distances agree; only the full graph differs, because its blocks
// are truncated to 8, 9 or 10 nodes.

TEST_F(AllPairsDistancesTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The ranks 0 and 2 use the same canonical node set, and so do the ranks 1 and 5 and the ranks
    // 3 and 6, but get_position_offset() translates the second copy each time, so no two nodes
    // coincide and the smallest distance is a positive one.
    // The values serve as standard for the seven-rank test
    expect_pair_distances(graph, 10.481250507373600, 0.042426406871192840, 24.867500879662195);
}

TEST_F(AllPairsDistancesTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_pair_distances(graph, 10.481250507373600, 0.042426406871192840, 24.867500879662195);
}

TEST_F(AllPairsDistancesTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_pair_distances(graph, 10.149053785805933, 0.042426406871192840, 24.867500879662195);
}

TEST_F(AllPairsDistancesTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version. With an odd number of ranks
    // every rank compares itself against the next three ones, which visits every pair exactly once
    expect_pair_distances(graph, 10.481250507373600, 0.042426406871192840, 24.867500879662195);
}

TEST_F(AllPairsDistancesTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_pair_distances(graph, 10.481250507373600, 0.042426406871192840, 24.867500879662195);
}

TEST_F(AllPairsDistancesTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_pair_distances(graph, 10.149053785805933, 0.042426406871192840, 24.867500879662195);
}
