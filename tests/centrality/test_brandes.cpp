/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_brandes.h"

#include "Types.h"

#include "metrics/paths/BetweennessCentrality.h"

#include <mpi-wrapper/core/MPIInfo.h>

namespace {
constexpr auto algorithm = BetweennessCentralityAlgorithm::Brandes;
} // namespace

TEST_F(BetweennessBrandesTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 13.583333333333332, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 5.4, 1e-6);
}

TEST_F(BetweennessBrandesTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 112.12194180319423, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 58.6, 1e-6);
}

TEST_F(BetweennessBrandesTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 112.12194180319423, 1e-6);
    }
}

TEST_F(BetweennessBrandesTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 58.6, 1e-6);
    }
}

TEST_F(BetweennessBrandesTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
    }
}

TEST_F(BetweennessBrandesTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // Only the six standard blocks contribute; the four arcs of the source/sink block are direct and
    // put no node in between. The value serves as standard for the seven-rank test
    ASSERT_NEAR(average_betweenness_centrality, 74.447050803480930, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The value serves as standard for the seven-rank test
    ASSERT_NEAR(average_betweenness_centrality, 39.171428571428571, 1e-6);
}

TEST_F(BetweennessBrandesTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    // The graph is complete, so every pair is joined by a single arc and no node lies in between.
    // The value serves as standard for the seven-rank test
    ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
}

TEST_F(BetweennessBrandesTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 74.447050803480930, 1e-6);
    }
}

TEST_F(BetweennessBrandesTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 39.171428571428571, 1e-6);
    }
}

TEST_F(BetweennessBrandesTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph, algorithm);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
    }
}
