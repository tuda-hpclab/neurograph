/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_louvain.h"

#include "Types.h"

#include "metrics/community/Modularity.h"

namespace {
constexpr auto algorithm = ModularityAlgorithm::Louvain;
} // namespace

TEST_F(ModularityLouvainTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value comes from a Python reference of the same deterministic algorithm; a brute force
    // over all 115975 partitions of the graph confirms that it is the optimal modularity.
    ASSERT_NEAR(modularity, 0.3180246913580247, 1e-6);
}

TEST_F(ModularityLouvainTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value comes from a Python reference of the same deterministic algorithm; a brute force
    // over all 115975 partitions of the graph confirms that it is the optimal modularity.
    ASSERT_NEAR(modularity, 0.26756198347107446, 1e-6);
}

TEST_F(ModularityLouvainTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // In the complete graph every partition has modularity zero: for every set of nodes, the
    // weight fraction inside equals the expected fraction from the strengths.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityLouvainTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test
    ASSERT_NEAR(modularity, 0.6626847034339229, 1e-6);
}

TEST_F(ModularityLouvainTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test
    ASSERT_NEAR(modularity, 0.45782470703125, 1e-6);
}

TEST_F(ModularityLouvainTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // In the complete graph every partition has modularity zero (see testFull)
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityLouvainTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.6626847034339229, 1e-6);
}

TEST_F(ModularityLouvainTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.45782470703125, 1e-6);
}

TEST_F(ModularityLouvainTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}
