/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree_counter.h"

#include "metrics/degree/AverageDegree.h"

#include <array>
#include <iostream>

TEST_F(AverageDegreeTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The values come from manually counting; total weight is 45, total number of arcs is 25
    ASSERT_NEAR(average_in_in_by_weights, 4.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 6.06666666666666, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 8.5111111111111111, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 4.0, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 7.2, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 10.92, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 15.32, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 7.2, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The graph is unweighted, so dividing by the weights and by the number of arcs yields the same value
    ASSERT_NEAR(average_in_in_by_weights, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 4.590909090909090909090, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 4.590909090909090909090, 1e-6);
}

TEST_F(AverageDegreeTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The graph is unweighted, so dividing by the weights and by the number of arcs yields the same value
    ASSERT_NEAR(average_in_in_by_weights, 10.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 10.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 10.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 10.0, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 10.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 10.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 10.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 10.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The values serve as standard for the four-rank test; total weight is 620, total number of arcs is 140
    ASSERT_NEAR(average_in_in_by_weights, 15.354838709677420, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 15.954838709677420, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 16.664516129032258, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 15.354838709677420, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 68.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 70.657142857142859, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 73.8, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 68.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The graph is unweighted, so dividing by the weights and by the number of arcs yields the same value
    ASSERT_NEAR(average_in_in_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 6.53125, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 6.53125, 1e-6);
}

TEST_F(AverageDegreeTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The graph is unweighted, so dividing by the weights and by the number of arcs yields the same value
    ASSERT_NEAR(average_in_in_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 35.0, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 35.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in_by_weights, 15.354838709677420, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 15.954838709677420, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 16.664516129032258, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 15.354838709677420, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 68.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 70.657142857142859, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 73.8, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 68.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 6.53125, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 6.53125, 1e-6);
}

TEST_F(AverageDegreeTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_weights, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_weights, 35.0, 1e-6);

    ASSERT_NEAR(average_in_in_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in_by_arcs, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out_by_arcs, 35.0, 1e-6);
}

TEST_F(AverageDegreeTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto average_in_in_by_weights = AverageDegree::all_compute_average_in_in_degree_by_weights(graph);
    const auto average_in_out_by_weights = AverageDegree::all_compute_average_in_out_degree_by_weights(graph);
    const auto average_out_in_by_weights = AverageDegree::all_compute_average_out_in_degree_by_weights(graph);
    const auto average_out_out_by_weights = AverageDegree::all_compute_average_out_out_degree_by_weights(graph);

    const auto average_in_in_by_arcs = AverageDegree::all_compute_average_in_in_degree_by_arcs(graph);
    const auto average_in_out_by_arcs = AverageDegree::all_compute_average_in_out_degree_by_arcs(graph);
    const auto average_out_in_by_arcs = AverageDegree::all_compute_average_out_in_degree_by_arcs(graph);
    const auto average_out_out_by_arcs = AverageDegree::all_compute_average_out_out_degree_by_arcs(graph);

    // A graph without arcs has no average degree; the guard against division by zero must yield 0.0 everywhere
    ASSERT_DOUBLE_EQ(average_in_in_by_weights, 0.0);
    ASSERT_DOUBLE_EQ(average_in_out_by_weights, 0.0);
    ASSERT_DOUBLE_EQ(average_out_in_by_weights, 0.0);
    ASSERT_DOUBLE_EQ(average_out_out_by_weights, 0.0);

    ASSERT_DOUBLE_EQ(average_in_in_by_arcs, 0.0);
    ASSERT_DOUBLE_EQ(average_in_out_by_arcs, 0.0);
    ASSERT_DOUBLE_EQ(average_out_in_by_arcs, 0.0);
    ASSERT_DOUBLE_EQ(average_out_out_by_arcs, 0.0);
}

namespace {
/**
 * @brief Checks the eight weighted average degrees. All of them are collective and return their
 *		result on every rank, so no rank is skipped here.
 * @param graph The graph
 * @param expected_by_weights The expected in-in, in-out, out-in and out-out average divided by the weights
 * @param expected_by_arcs The expected in-in, in-out, out-in and out-out average divided by the arc counts
 */
void expect_average_degrees(const DistributedGraph& graph, const std::array<double, 4>& expected_by_weights,
                            const std::array<double, 4>& expected_by_arcs) {
    ASSERT_NEAR(AverageDegree::all_compute_average_in_in_degree_by_weights(graph), expected_by_weights[0], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_in_out_degree_by_weights(graph), expected_by_weights[1], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_out_in_degree_by_weights(graph), expected_by_weights[2], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_out_out_degree_by_weights(graph), expected_by_weights[3], 1e-6);

    ASSERT_NEAR(AverageDegree::all_compute_average_in_in_degree_by_arcs(graph), expected_by_arcs[0], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_in_out_degree_by_arcs(graph), expected_by_arcs[1], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_out_in_degree_by_arcs(graph), expected_by_arcs[2], 1e-6);
    ASSERT_NEAR(AverageDegree::all_compute_average_out_out_degree_by_arcs(graph), expected_by_arcs[3], 1e-6);
}
} // namespace

TEST_F(AverageDegreeTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The total weight is 940 and the total number of arcs is 214.
    // The values serve as standard for the seven-rank test
    expect_average_degrees(graph,
                           { 15.191489361702128, 15.817021276595744, 16.527659574468085, 15.191489361702128 },
                           { 66.728971962616825, 69.476635514018690, 72.598130841121490, 66.728971962616825 });
}

TEST_F(AverageDegreeTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The graph is unweighted, so dividing by the weights and by the number of arcs yields the same value.
    // The values serve as standard for the seven-rank test
    expect_average_degrees(graph, { 6.1344086021505375, 6.1344086021505375, 6.1344086021505375, 6.1344086021505375 },
                           { 6.1344086021505375, 6.1344086021505375, 6.1344086021505375, 6.1344086021505375 });
}

TEST_F(AverageDegreeTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // Every node has 62 in and 62 out arcs, so all eight averages are 62.
    // The values serve as standard for the seven-rank test
    expect_average_degrees(graph, { 62.0, 62.0, 62.0, 62.0 }, { 62.0, 62.0, 62.0, 62.0 });
}

TEST_F(AverageDegreeTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_average_degrees(graph,
                           { 15.191489361702128, 15.817021276595744, 16.527659574468085, 15.191489361702128 },
                           { 66.728971962616825, 69.476635514018690, 72.598130841121490, 66.728971962616825 });
}

TEST_F(AverageDegreeTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_average_degrees(graph, { 6.1344086021505375, 6.1344086021505375, 6.1344086021505375, 6.1344086021505375 },
                           { 6.1344086021505375, 6.1344086021505375, 6.1344086021505375, 6.1344086021505375 });
}

TEST_F(AverageDegreeTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_average_degrees(graph, { 62.0, 62.0, 62.0, 62.0 }, { 62.0, 62.0, 62.0, 62.0 });
}
