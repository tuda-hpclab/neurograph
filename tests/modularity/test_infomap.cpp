/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_infomap.h"

#include "Types.h"

#include "metrics/community/Modularity.h"

namespace {
constexpr auto algorithm = ModularityAlgorithm::Infomap;
} // namespace

// Infomap minimizes the description length of a random walk instead of maximizing the modularity,
// so its partitions differ from the ones of Louvain and Leiden and their modularity is at most as
// high as theirs. All values come from a Python reference of the same deterministic algorithm, which
// reproduces the committed Louvain and Leiden values as well; on the graphs with ten nodes a brute
// force over all 115975 partitions confirms that the found partition has the shortest description.

TEST_F(ModularityInfomapTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The walk is described with the fewest bits by the two modules {3, 5, 7} and the other seven
    // nodes (2.927 instead of 3.026 bits for one module). That partition is coarser than the four
    // communities of Louvain and Leiden, so its modularity stays below their 0.318.
    ASSERT_NEAR(modularity, 0.1679012345679012, 1e-6);
}

TEST_F(ModularityInfomapTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // Here the two communities of Louvain and Leiden do not pay off for the walk: describing it
    // with one module costs 3.3017 bits and with those two modules 3.3020, so everything is merged
    // and a single community always has modularity zero.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityInfomapTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // In the complete graph the walk visits every node equally often, so no module can compress it;
    // besides, every partition of the complete graph has modularity zero.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityInfomapTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // A graph with a source and a sink node, which is what makes the teleportation of the walker
    // necessary: without it the whole flow would pile up in the sink. The source node 0 becomes a
    // module of its own, which happens to have modularity zero on this graph.
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityInfomapTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test. The ten modules of four nodes stay below
    // the 0.663 of Louvain and Leiden, which merge them into larger communities.
    ASSERT_NEAR(modularity, 0.6299687825182102, 1e-6);
}

TEST_F(ModularityInfomapTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value serves as standard for the four-rank test
    ASSERT_NEAR(modularity, 0.451324462890625, 1e-6);
}

TEST_F(ModularityInfomapTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // No module compresses the walk on the complete graph (see testFull)
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}

TEST_F(ModularityInfomapTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.6299687825182102, 1e-6);
}

TEST_F(ModularityInfomapTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.451324462890625, 1e-6);
}

TEST_F(ModularityInfomapTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto modularity = Modularity::compute_modularity(graph, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_NEAR(modularity, 0.0, 1e-6);
}
