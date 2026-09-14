/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_multistep.h"

#include "Types.h"

#include "metrics/connectivity/StronglyConnectedComponents.h"

#include <unordered_map>

// All algorithms find the same components, so the expected values are the ones derived in
// test_tarjan_gathered.cpp, which computes them with the sequential reference algorithm.

namespace {
constexpr auto algorithm = SccAlgorithm::Multistep;

using SizeHistogram = std::unordered_map<global_node_id_type, global_node_id_type>;

// Enough to hand every node of the test graphs to the root rank, which exercises the last phase
constexpr auto gathering_tail_threshold = global_node_id_type{ 64 };
} // namespace

TEST_F(SccMultistepTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Nothing can be trimmed from a strongly connected graph, and the forward-backward round then
    // peels it off as a whole, so no coloring round is left to run
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccMultistepTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccMultistepTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccMultistepTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Without nodes there is nothing for any of the phases to do
    ASSERT_EQ(result.number_components, 0);
    ASSERT_EQ(result.largest_component_size, 0);
    ASSERT_EQ(result.component_sizes, SizeHistogram{});
}

TEST_F(SccMultistepTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The best case for this variant: node 0 has no in arc and node 3 has no out arc, so trimming
    // removes the whole graph and neither the forward-backward round nor a coloring round is needed
    ASSERT_EQ(result.number_components, 4);
    ASSERT_EQ(result.largest_component_size, 1);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 } }));
}

TEST_F(SccMultistepTest, testMultiComponentFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The graph exercises all distributed phases: trimming removes the nodes 1 and 2 one after the
    // other, the forward-backward round peels the pair 4 <-> 5 off, and the coloring rounds find the
    // ring and the remaining components
    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccMultistepTest, testMultiComponentFourRanksDummyWithGatheringTail) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm, gathering_tail_threshold);

    // With a threshold this large the coloring rounds never run: whatever trimming and the
    // forward-backward round left over is gathered on the root rank, which must not change the result
    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccMultistepTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccMultistepTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 36);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 36, 1 } }));
}

TEST_F(SccMultistepTest, testMultiComponentFourRanks) {
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

TEST_F(SccMultistepTest, testMultiComponentFourRanksWithGatheringTail) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm, gathering_tail_threshold);

    // Gathering the rest on the root rank must not change the result, and every rank still receives it
    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccMultistepTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccMultistepTest, testFullFourRanks) {
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

TEST_F(SccMultistepTest, testStandardSevenRanksDummy) {
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

TEST_F(SccMultistepTest, testStandardUUSevenRanksDummy) {
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

TEST_F(SccMultistepTest, testFullSevenRanksDummy) {
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

TEST_F(SccMultistepTest, testStandardSevenRanks) {
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

TEST_F(SccMultistepTest, testStandardUUSevenRanks) {
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

TEST_F(SccMultistepTest, testFullSevenRanks) {
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
