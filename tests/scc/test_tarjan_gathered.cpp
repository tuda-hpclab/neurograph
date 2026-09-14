/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_tarjan_gathered.h"

#include "Types.h"

#include "metrics/connectivity/StronglyConnectedComponents.h"

#include <unordered_map>

namespace {
constexpr auto algorithm = SccAlgorithm::TarjanGathered;

using SizeHistogram = std::unordered_map<global_node_id_type, global_node_id_type>;
} // namespace

TEST_F(SccTarjanGatheredTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The standard graph is strongly connected: the cycle 0 -> 6 -> 9 -> 1 -> 0 carries the nodes 0, 1,
    // 6, and 9, and every other node lies on a cycle with them as well, e.g. 4 through 9 -> 4 -> 1 and
    // 5 through 3 -> 5 -> 7 -> 3
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Synchronizing the arcs of a strongly connected graph cannot separate anything, so the symmetric
    // graph is one component as well
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // In the complete graph every node reaches every other one directly. Its self arcs are ignored,
    // which does not change the component: a self arc joins no two nodes
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 10);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 10, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Without nodes there is nothing that could form a component
    ASSERT_EQ(result.number_components, 0);
    ASSERT_EQ(result.largest_component_size, 0);
    ASSERT_EQ(result.component_sizes, SizeHistogram{});
}

TEST_F(SccTarjanGatheredTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The arcs 0 -> 1, 0 -> 2, 1 -> 2, 1 -> 3, and 2 -> 3 all point from a smaller to a larger node, so
    // the graph has no cycle at all and every node is a component of its own
    ASSERT_EQ(result.number_components, 4);
    ASSERT_EQ(result.largest_component_size, 1);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 } }));
}

TEST_F(SccTarjanGatheredTest, testMultiComponentFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_multi_component_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The twelve nodes form the ring 0 -> 3 -> 6 -> 9 -> 0, the pairs 4 <-> 5 and 7 <-> 10, and the
    // four single nodes 1, 2, 8, and 11, which lie on the paths between the other components.
    // The values serve as standard for the four-rank test
    ASSERT_EQ(result.number_components, 7);
    ASSERT_EQ(result.largest_component_size, 4);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 1, 4 }, { 2, 2 }, { 4, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // Each of the four copies of the standard graph is strongly connected on its own, and the ring arcs
    // join the copies into a cycle, so all 40 nodes end up in one component.
    // The values serve as standard for the four-rank test
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 36);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 36, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testMultiComponentFourRanks) {
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

TEST_F(SccTarjanGatheredTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 40);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 40, 1 } }));
}

TEST_F(SccTarjanGatheredTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto result = StronglyConnectedComponents::compute_strongly_connected_components(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(result.number_components, 1);
    ASSERT_EQ(result.largest_component_size, 36);
    ASSERT_EQ(result.component_sizes, (SizeHistogram{ { 36, 1 } }));
}

// The seven-rank graphs fall apart into three parts: the four blocks of the large ring, the two
// blocks of the small one, and the source/sink block, whose arcs all lie on no cycle.

TEST_F(SccTarjanGatheredTest, testStandardSevenRanksDummy) {
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

TEST_F(SccTarjanGatheredTest, testStandardUUSevenRanksDummy) {
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

TEST_F(SccTarjanGatheredTest, testFullSevenRanksDummy) {
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

TEST_F(SccTarjanGatheredTest, testStandardSevenRanks) {
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

TEST_F(SccTarjanGatheredTest, testStandardUUSevenRanks) {
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

TEST_F(SccTarjanGatheredTest, testFullSevenRanks) {
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
