/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree.h"

#include "metrics/degree/InDegree.h"
#include "metrics/degree/OutDegree.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(DegreeTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 1);
    ASSERT_EQ(maximum_in_1, 5);
    ASSERT_EQ(minimum_in_2, 1);
    ASSERT_EQ(maximum_in_2, 5);

    ASSERT_EQ(all_minimum_in_1, 1);
    ASSERT_EQ(all_maximum_in_1, 5);
    ASSERT_EQ(all_minimum_in_2, 1);
    ASSERT_EQ(all_maximum_in_2, 5);

    ASSERT_EQ(minimum_out_1, 2);
    ASSERT_EQ(maximum_out_1, 4);
    ASSERT_EQ(minimum_out_2, 2);
    ASSERT_EQ(maximum_out_2, 4);

    ASSERT_EQ(all_minimum_out_1, 2);
    ASSERT_EQ(all_maximum_out_1, 4);
    ASSERT_EQ(all_minimum_out_2, 2);
    ASSERT_EQ(all_maximum_out_2, 4);
}

TEST_F(DegreeTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 3);
    ASSERT_EQ(maximum_in_1, 6);
    ASSERT_EQ(minimum_in_2, 3);
    ASSERT_EQ(maximum_in_2, 6);

    ASSERT_EQ(all_minimum_in_1, 3);
    ASSERT_EQ(all_maximum_in_1, 6);
    ASSERT_EQ(all_minimum_in_2, 3);
    ASSERT_EQ(all_maximum_in_2, 6);

    ASSERT_EQ(minimum_out_1, 3);
    ASSERT_EQ(maximum_out_1, 6);
    ASSERT_EQ(minimum_out_2, 3);
    ASSERT_EQ(maximum_out_2, 6);

    ASSERT_EQ(all_minimum_out_1, 3);
    ASSERT_EQ(all_maximum_out_1, 6);
    ASSERT_EQ(all_minimum_out_2, 3);
    ASSERT_EQ(all_maximum_out_2, 6);
}

TEST_F(DegreeTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 10);
    ASSERT_EQ(maximum_in_1, 10);
    ASSERT_EQ(minimum_in_2, 10);
    ASSERT_EQ(maximum_in_2, 10);

    ASSERT_EQ(all_minimum_in_1, 10);
    ASSERT_EQ(all_maximum_in_1, 10);
    ASSERT_EQ(all_minimum_in_2, 10);
    ASSERT_EQ(all_maximum_in_2, 10);

    ASSERT_EQ(minimum_out_1, 10);
    ASSERT_EQ(maximum_out_1, 10);
    ASSERT_EQ(minimum_out_2, 10);
    ASSERT_EQ(maximum_out_2, 10);

    ASSERT_EQ(all_minimum_out_1, 10);
    ASSERT_EQ(all_maximum_out_1, 10);
    ASSERT_EQ(all_minimum_out_2, 10);
    ASSERT_EQ(all_maximum_out_2, 10);
}

TEST_F(DegreeTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 2);
    ASSERT_EQ(maximum_in_1, 6);
    ASSERT_EQ(minimum_in_2, 2);
    ASSERT_EQ(maximum_in_2, 6);

    ASSERT_EQ(all_minimum_in_1, 2);
    ASSERT_EQ(all_maximum_in_1, 6);
    ASSERT_EQ(all_minimum_in_2, 2);
    ASSERT_EQ(all_maximum_in_2, 6);

    ASSERT_EQ(minimum_out_1, 3);
    ASSERT_EQ(maximum_out_1, 5);
    ASSERT_EQ(minimum_out_2, 3);
    ASSERT_EQ(maximum_out_2, 5);

    ASSERT_EQ(all_minimum_out_1, 3);
    ASSERT_EQ(all_maximum_out_1, 5);
    ASSERT_EQ(all_minimum_out_2, 3);
    ASSERT_EQ(all_maximum_out_2, 5);
}

TEST_F(DegreeTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 5);
    ASSERT_EQ(maximum_in_1, 8);
    ASSERT_EQ(minimum_in_2, 5);
    ASSERT_EQ(maximum_in_2, 8);

    ASSERT_EQ(all_minimum_in_1, 5);
    ASSERT_EQ(all_maximum_in_1, 8);
    ASSERT_EQ(all_minimum_in_2, 5);
    ASSERT_EQ(all_maximum_in_2, 8);

    ASSERT_EQ(minimum_out_1, 5);
    ASSERT_EQ(maximum_out_1, 8);
    ASSERT_EQ(minimum_out_2, 5);
    ASSERT_EQ(maximum_out_2, 8);

    ASSERT_EQ(all_minimum_out_1, 5);
    ASSERT_EQ(all_maximum_out_1, 8);
    ASSERT_EQ(all_minimum_out_2, 5);
    ASSERT_EQ(all_maximum_out_2, 8);
}

TEST_F(DegreeTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 35);
    ASSERT_EQ(maximum_in_1, 35);
    ASSERT_EQ(minimum_in_2, 35);
    ASSERT_EQ(maximum_in_2, 35);

    ASSERT_EQ(all_minimum_in_1, 35);
    ASSERT_EQ(all_maximum_in_1, 35);
    ASSERT_EQ(all_minimum_in_2, 35);
    ASSERT_EQ(all_maximum_in_2, 35);

    ASSERT_EQ(minimum_out_1, 35);
    ASSERT_EQ(maximum_out_1, 35);
    ASSERT_EQ(minimum_out_2, 35);
    ASSERT_EQ(maximum_out_2, 35);

    ASSERT_EQ(all_minimum_out_1, 35);
    ASSERT_EQ(all_maximum_out_1, 35);
    ASSERT_EQ(all_minimum_out_2, 35);
    ASSERT_EQ(all_maximum_out_2, 35);
}

TEST_F(DegreeTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 2);
        ASSERT_EQ(maximum_in_1, 6);
        ASSERT_EQ(minimum_in_2, 2);
        ASSERT_EQ(maximum_in_2, 6);
    }

    ASSERT_EQ(all_minimum_in_1, 2);
    ASSERT_EQ(all_maximum_in_1, 6);
    ASSERT_EQ(all_minimum_in_2, 2);
    ASSERT_EQ(all_maximum_in_2, 6);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 3);
        ASSERT_EQ(maximum_out_1, 5);
        ASSERT_EQ(minimum_out_2, 3);
        ASSERT_EQ(maximum_out_2, 5);
    }

    ASSERT_EQ(all_minimum_out_1, 3);
    ASSERT_EQ(all_maximum_out_1, 5);
    ASSERT_EQ(all_minimum_out_2, 3);
    ASSERT_EQ(all_maximum_out_2, 5);
}

TEST_F(DegreeTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 5);
        ASSERT_EQ(maximum_in_1, 8);
        ASSERT_EQ(minimum_in_2, 5);
        ASSERT_EQ(maximum_in_2, 8);
    }

    ASSERT_EQ(all_minimum_in_1, 5);
    ASSERT_EQ(all_maximum_in_1, 8);
    ASSERT_EQ(all_minimum_in_2, 5);
    ASSERT_EQ(all_maximum_in_2, 8);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 5);
        ASSERT_EQ(maximum_out_1, 8);
        ASSERT_EQ(minimum_out_2, 5);
        ASSERT_EQ(maximum_out_2, 8);
    }

    ASSERT_EQ(all_minimum_out_1, 5);
    ASSERT_EQ(all_maximum_out_1, 8);
    ASSERT_EQ(all_minimum_out_2, 5);
    ASSERT_EQ(all_maximum_out_2, 8);
}

TEST_F(DegreeTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto minimum_in_1 = InDegree::compute_minimum_degree(graph);
    const auto maximum_in_1 = InDegree::compute_maximum_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = InDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in_1 = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in_1 = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = InDegree::all_compute_extreme_degrees(graph);

    const auto minimum_out_1 = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out_1 = OutDegree::compute_maximum_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_out_1 = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out_1 = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = OutDegree::all_compute_extreme_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 35);
        ASSERT_EQ(maximum_in_1, 35);
        ASSERT_EQ(minimum_in_2, 35);
        ASSERT_EQ(maximum_in_2, 35);
    }

    ASSERT_EQ(all_minimum_in_1, 35);
    ASSERT_EQ(all_maximum_in_1, 35);
    ASSERT_EQ(all_minimum_in_2, 35);
    ASSERT_EQ(all_maximum_in_2, 35);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 35);
        ASSERT_EQ(maximum_out_1, 35);
        ASSERT_EQ(minimum_out_2, 35);
        ASSERT_EQ(maximum_out_2, 35);
    }

    ASSERT_EQ(all_minimum_out_1, 35);
    ASSERT_EQ(all_maximum_out_1, 35);
    ASSERT_EQ(all_minimum_out_2, 35);
    ASSERT_EQ(all_maximum_out_2, 35);
}

namespace {
/**
 * @brief Checks all six degree entry points against the expected extremes. The collective variants
 *		return the result on every rank, the plain ones only on the root rank.
 * @param graph The graph
 * @param expected_minimum_in The expected smallest in degree
 * @param expected_maximum_in The expected largest in degree
 * @param expected_minimum_out The expected smallest out degree
 * @param expected_maximum_out The expected largest out degree
 */
void expect_extreme_degrees(const DistributedGraph& graph, const arc_id_type expected_minimum_in, const arc_id_type expected_maximum_in,
                            const arc_id_type expected_minimum_out, const arc_id_type expected_maximum_out) {
    const auto minimum_in = InDegree::compute_minimum_degree(graph);
    const auto maximum_in = InDegree::compute_maximum_degree(graph);
    const auto& [extreme_minimum_in, extreme_maximum_in] = InDegree::compute_extreme_degrees(graph);

    const auto minimum_out = OutDegree::compute_minimum_degree(graph);
    const auto maximum_out = OutDegree::compute_maximum_degree(graph);
    const auto& [extreme_minimum_out, extreme_maximum_out] = OutDegree::compute_extreme_degrees(graph);

    const auto all_minimum_in = InDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_in = InDegree::all_compute_maximum_degree(graph);
    const auto& [all_extreme_minimum_in, all_extreme_maximum_in] = InDegree::all_compute_extreme_degrees(graph);

    const auto all_minimum_out = OutDegree::all_compute_minimum_degree(graph);
    const auto all_maximum_out = OutDegree::all_compute_maximum_degree(graph);
    const auto& [all_extreme_minimum_out, all_extreme_maximum_out] = OutDegree::all_compute_extreme_degrees(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in, expected_minimum_in);
        ASSERT_EQ(maximum_in, expected_maximum_in);
        ASSERT_EQ(extreme_minimum_in, expected_minimum_in);
        ASSERT_EQ(extreme_maximum_in, expected_maximum_in);

        ASSERT_EQ(minimum_out, expected_minimum_out);
        ASSERT_EQ(maximum_out, expected_maximum_out);
        ASSERT_EQ(extreme_minimum_out, expected_minimum_out);
        ASSERT_EQ(extreme_maximum_out, expected_maximum_out);
    }

    ASSERT_EQ(all_minimum_in, expected_minimum_in);
    ASSERT_EQ(all_maximum_in, expected_maximum_in);
    ASSERT_EQ(all_extreme_minimum_in, expected_minimum_in);
    ASSERT_EQ(all_extreme_maximum_in, expected_maximum_in);

    ASSERT_EQ(all_minimum_out, expected_minimum_out);
    ASSERT_EQ(all_maximum_out, expected_maximum_out);
    ASSERT_EQ(all_extreme_minimum_out, expected_minimum_out);
    ASSERT_EQ(all_extreme_maximum_out, expected_maximum_out);
}
} // namespace

TEST_F(DegreeTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The isolated nodes of the source/sink block push both minima down to zero; the maxima are the
    // ones of the standard block plus the single ring arc.
    // The values serve as standard for the seven-rank test
    expect_extreme_degrees(graph, 0, 6, 0, 5);
}

TEST_F(DegreeTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_extreme_degrees(graph, 0, 8, 0, 8);
}

TEST_F(DegreeTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The complete graph is 62-regular, so all four extremes agree.
    // The values serve as standard for the seven-rank test
    expect_extreme_degrees(graph, 62, 62, 62, 62);
}

TEST_F(DegreeTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_extreme_degrees(graph, 0, 6, 0, 5);
}

TEST_F(DegreeTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_extreme_degrees(graph, 0, 8, 0, 8);
}

TEST_F(DegreeTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_extreme_degrees(graph, 62, 62, 62, 62);
}
