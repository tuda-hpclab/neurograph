/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_length.h"

#include "metrics/geometry/ArcLength.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(ArcLengthTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    // A graph without arcs has no average arc length; the functions must return 0.0 instead of dividing by zero.
    ASSERT_DOUBLE_EQ(ArcLength::compute_average_arc_length(graph), 0.0);
    ASSERT_DOUBLE_EQ(ArcLength::all_compute_average_arc_length(graph), 0.0);
}

TEST_F(ArcLengthTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 5.646332800503403, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 10.860479, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 5.646332800503403, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 10.860479, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 6.472791682288855, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 10.860479, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 6.472791682288855, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 10.860479, 1e-6);
}

TEST_F(ArcLengthTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 6.662768699459406, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 11.969126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 6.662768699459406, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 11.969126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 4.0593813369217930, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 4.0593813369217930, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.544612, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.544612, 1e-6);
}

TEST_F(ArcLengthTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 4.0593813369217930, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 4.0593813369217930, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 2.998320428459754, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 4.0564215744513206, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.544612, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.544612, 1e-6);
}

namespace {
/**
 * @brief Checks all six arc length entry points against the expected values. The plain variants
 *		return their result only on the root rank, the collective ones on every rank.
 * @param graph The graph
 * @param expected_average The expected weighted average arc length
 * @param expected_minimum The expected shortest arc
 * @param expected_maximum The expected longest arc
 */
void expect_arc_lengths(const DistributedGraph& graph, const double expected_average, const double expected_minimum,
                        const double expected_maximum) {
    // All six are collectives, so they run before the first assertion: a failing assertion returns
    // early, and a rank that skipped a collective would leave the others waiting forever
    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_NEAR(average_arc_length, expected_average, 1e-6);
        ASSERT_NEAR(minimum_arc_length, expected_minimum, 1e-6);
        ASSERT_NEAR(maximum_arc_length, expected_maximum, 1e-6);
    }

    ASSERT_NEAR(all_average_arc_length, expected_average, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, expected_minimum, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, expected_maximum, 1e-6);
}
} // namespace

TEST_F(ArcLengthTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // Ranks 2, 5, and 6 hold their node sets twelve units away from the origin, so the ring arcs
    // into and out of them are long. The shortest arc is still a self arc of length zero.
    // The values serve as standard for the seven-rank test
    expect_arc_lengths(graph, 8.8532913353555520, 0.0, 24.322773279377497);
}

TEST_F(ArcLengthTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_arc_lengths(graph, 4.5973183618887640, 0.0, 24.322773279377497);
}

TEST_F(ArcLengthTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The complete graph has no self arc, so the shortest arc is the one between the two closest
    // distinct nodes, and the longest one connects the two most distant ones.
    // The values serve as standard for the seven-rank test
    expect_arc_lengths(graph, 10.149053785805933, 0.042426406871192840, 24.867500879662195);
}

TEST_F(ArcLengthTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_arc_lengths(graph, 8.8532913353555520, 0.0, 24.322773279377497);
}

TEST_F(ArcLengthTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_arc_lengths(graph, 4.5973183618887640, 0.0, 24.322773279377497);
}

TEST_F(ArcLengthTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_arc_lengths(graph, 10.149053785805933, 0.042426406871192840, 24.867500879662195);
}
