/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_label_propagation.h"

#include "Types.h"

#include "metrics/community/Modularity.h"

namespace {
constexpr auto algorithm = ModularityAlgorithm::LabelPropagation;
} // namespace

// Label propagation maximizes nothing, so its partitions differ from the ones of Louvain and Leiden
// and its modularity is at most as high as theirs. All values come from a Python reference of the
// same deterministic algorithm, which reproduces the committed Louvain and Leiden values as well.

TEST_F(ModularityLabelPropagationTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The label of node 6, which carries the most weight, spreads over the whole graph within five
    // sweeps. A single community always has modularity zero, so nothing of the structure that
    // Louvain and Leiden find (four communities, modularity 0.318) is left.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // On the symmetric graph with unit weights the propagation stops at the same two communities
    // that Louvain and Leiden find, so it reaches their (brute-force optimal) modularity
    ASSERT_NEAR(modularity, 0.26756198347107446, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // In the complete graph every partition has modularity zero: for every set of nodes, the
    // weight fraction inside equals the expected fraction from the strengths.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // A graph with a source and a sink node, whose weights only count by absolute value. The
    // propagation splits it into {0, 1} and {2, 3}, which is what Louvain and Leiden find too.
    ASSERT_NEAR(modularity, 0.16666666666666666, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test. The ten communities of four nodes stay
    // below the 0.663 of Louvain and Leiden, which merge them into larger ones.
    ASSERT_NEAR(modularity, 0.6299687825182102, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test
    ASSERT_NEAR(modularity, 0.33544921875, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // One label takes over the complete graph, and a single community has modularity zero
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.6299687825182102, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.33544921875, 1e-6);
}

TEST_F(ModularityLabelPropagationTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}
