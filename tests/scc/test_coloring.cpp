/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_coloring.h"

#include "Types.h"

#include "metrics/connectivity/StronglyConnectedComponents.h"

#include <unordered_map>

// All algorithms find the same components, so the expected values are the ones derived in
// test_tarjan_gathered.cpp, which computes them with the sequential reference algorithm.

namespace {
constexpr auto algorithm = SccAlgorithm::Coloring;

using SizeHistogram = std::unordered_map<global_node_id_type, global_node_id_type>;
} // namespace

TEST_F(SccColoringTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The largest global id colors the whole graph, so its own component is the whole graph
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccColoringTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccColoringTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccColoringTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Without nodes there is nothing to color, so not a single round runs
    ASSERT_EQ(result.number_components, 0);
    ASSERT_EQ(result.largest_component_size, 0);
    ASSERT_EQ(result.component_sizes, SizeHistogram{});
}

TEST_F(SccColoringTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The best case for this variant: no node is colored by another one, so every node is a root and
    // one round finds all four components
    ASSERT_EQ(result.number_components, 4);
    ASSERT_EQ(result.largest_component_size, 1);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 } }));
}

TEST_F(SccColoringTest, testMultiComponentFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccColoringTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccColoringTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 36);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 36, 1 } }));
}

TEST_F(SccColoringTest, testMultiComponentFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version; every rank receives them
    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccColoringTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccColoringTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 36);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 36, 1 } }));
}

// The seven-rank graphs fall apart into three parts: the four blocks of the large ring, the two
// blocks of the small one, and the source/sink block, whose arcs all lie on no cycle.

TEST_F(SccColoringTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The two rings are strongly connected on their own; every node of the source/sink block is a
    // component of its own. The values serve as standard for the seven-rank test
    ASSERT_EQ(result.number_components, 12);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 10 }, { 20, 1 }, { 40, 1 } }));
}

TEST_F(SccColoringTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Symmetrizing merges the sinks and sources of the source/sink block into {3, 6, 9}, {4, 7} and
    // {5, 8}; only its three isolated nodes stay alone.
    // The values serve as standard for the seven-rank test
    ASSERT_EQ(result.number_components, 8);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 3 }, { 2, 2 }, { 3, 1 }, { 20, 1 }, { 40, 1 } }));
}

TEST_F(SccColoringTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values serve as standard for the seven-rank test
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 63);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 63, 1 } }));
}

TEST_F(SccColoringTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version; every rank receives them
    ASSERT_EQ(result.number_components, 12);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 10 }, { 20, 1 }, { 40, 1 } }));
}

TEST_F(SccColoringTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(result.number_components, 8);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 3 }, { 2, 2 }, { 3, 1 }, { 20, 1 }, { 40, 1 } }));
}

TEST_F(SccColoringTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 63);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 63, 1 } }));
}
