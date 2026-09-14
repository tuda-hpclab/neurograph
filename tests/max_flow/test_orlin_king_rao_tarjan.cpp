/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_orlin_king_rao_tarjan.h"

#include "Types.h"

#include "metrics/flow/MaximumFlow.h"

#include <vector>

namespace {
constexpr auto algorithm = MaxFlowAlgorithm::OrlinKingRaoTarjan;
} // namespace

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 6 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The minimum cut separates the source from the rest of the graph: its out arcs
    // 0 -> 6, 0 -> 8, and 0 -> 9 carry 1 + 2 + 1 = 4
    ASSERT_EQ(max_flow, 4);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardBottleneck) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 5 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 9 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The source can emit 11 and the sink can absorb 4, but the minimum cut lies in between:
    // the arcs 1 -> 0, 4 -> 8, and 6 -> 9 with capacity 1 + 1 + 1 = 3
    ASSERT_EQ(max_flow, 3);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardReverse) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto nodes_0 = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto nodes_9 = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 9 } };

    const auto max_flow_forward = MaximumFlow::compute_maximum_flow(graph, nodes_0, nodes_9, algorithm);
    const auto max_flow_backward = MaximumFlow::compute_maximum_flow(graph, nodes_9, nodes_0, algorithm);

    // The flow respects the arc directions: from 0 to 9 the out arcs of the source carry
    // 1 + 2 + 1 = 4, while from 9 to 0 everything must pass the single arc 1 -> 0 with capacity 1
    ASSERT_EQ(max_flow_forward, 4);
    ASSERT_EQ(max_flow_backward, 1);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardMultipleSourcesAndSinks) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 }, NodeIdentifier{ 0, 5 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 8 }, NodeIdentifier{ 0, 9 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The minimum cut consists of all arcs into the sinks from outside: 0 -> 8 (2), 4 -> 8 (1),
    // 0 -> 9 (1), and 6 -> 9 (1); the arc 8 -> 9 runs between the sinks and does not count
    ASSERT_EQ(max_flow, 5);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 5 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // In the undirected unit-weight graph the sink has the three neighbors 3, 6, and 7, and the
    // arc-disjoint paths 0-6-5, 0-1-3-5, and 0-8-3-7-5 saturate all three arcs into the sink
    ASSERT_EQ(max_flow, 3);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 9 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The complete unit-weight graph connects the source to the sink directly and through each
    // of the eight other nodes: 1 + 8 = 9
    ASSERT_EQ(max_flow, 9);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testNegativeWeights) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto nodes_0 = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto nodes_3 = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 3 } };

    const auto max_flow_forward = MaximumFlow::compute_maximum_flow(graph, nodes_0, nodes_3, algorithm);
    const auto max_flow_backward = MaximumFlow::compute_maximum_flow(graph, nodes_3, nodes_0, algorithm);

    // The weights enter by absolute value, so the minimum cut separates the source from the rest
    // of the graph: its out arcs carry |-3| + |2| = 5. Node 3 has no out arcs, so no flow runs backwards
    ASSERT_EQ(max_flow_forward, 5);
    ASSERT_EQ(max_flow_backward, 0);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 10 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value serves as standard for the four-rank test: the sink is only entered by the ring
    // arc from the source (11) and by its local in arc 11 -> 10 (1), and both are saturated
    ASSERT_EQ(max_flow, 12);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 10 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value serves as standard for the four-rank test: in the undirected unit-weight graph
    // the sink has six neighbors (its local neighbors 11, 16, 18, and 19 plus its ring neighbors
    // 0 and 20), and six arc-disjoint paths from the source exist
    ASSERT_EQ(max_flow, 6);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 8 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The complete unit-weight graph over all 36 nodes connects the source to the sink directly and
    // through each of the 34 other nodes: 1 + 34 = 35. Node 8 is the first node of rank 1, i.e. the
    // node that the four-rank test below addresses as NodeIdentifier{ 1, 0 }
    ASSERT_EQ(max_flow, 35);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 12);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 6);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 35);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardMultipleSourcesAndSinksFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 }, NodeIdentifier{ 0, 1 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 }, NodeIdentifier{ 1, 1 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The minimum cut consists of the two ring arcs into the sinks (11 each) and the local arcs
    // 3 -> 1, 9 -> 1, and 4 -> 1 on rank 1 (1 + 2 + 1); the arc from sink (1, 1) to sink (1, 0)
    // does not count: 22 + 4 = 26
    ASSERT_EQ(max_flow, 26);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 10 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // Source and sink sit in the four-block ring, whose arcs are the same as in the four-rank graph,
    // so the minimum cut is the same one: the ring arc into the sink (11) and its local in arc (1).
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(max_flow, 12);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 10 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(max_flow, 6);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // Node 8 is the first node of the second block, i.e. the node (1, 0) of the seven-rank version
    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 8 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The complete graph connects the source to the sink directly and through each of the 61 other
    // nodes: 1 + 61 = 62. The value serves as standard for the seven-rank test
    ASSERT_EQ(max_flow, 62);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardSeparateComponentsSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // Node 20 is the first node of the two-block ring, i.e. the node (2, 0) of the seven-rank version
    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 20 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // No arc leaves the four-block ring, so nothing can reach the sink.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(max_flow, 0);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 12);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 6);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 1, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 62);
}

TEST_F(MaxFlowOrlinKingRaoTarjanTest, testStandardSeparateComponentsSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto sources = std::vector<NodeIdentifier>{ NodeIdentifier{ 0, 0 } };
    const auto sinks = std::vector<NodeIdentifier>{ NodeIdentifier{ 2, 0 } };

    const auto max_flow = MaximumFlow::compute_maximum_flow(graph, sources, sinks, algorithm);

    // The value is the corresponding one from the one-rank version
    ASSERT_EQ(max_flow, 0);
}
