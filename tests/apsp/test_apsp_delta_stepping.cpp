/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_apsp_delta_stepping.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/paths/AllPairsShortestPath.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <string>
#include <utility>
#include <vector>

// compute_apsp_delta_stepping solves the same problem as compute_apsp, so the expectations below are
// the values that test_apsp.cpp asserts for the Dijkstra-based computation on the same graph; the two
// have to agree exactly. What differs is the way there: delta-stepping is itself collective, so every
// rank walks all global sources together instead of only its own nodes, and the per-source results are
// already reduced across the ranks when they arrive, which is why the aggregation performs no further
// reduction of its own. A result whose sums are those of the graph times the rank count would be one
// that reduced them a second time, which is what the distributed tests below rule out.
//
// The five epoch strategies are the test parameter. The delta is not: the tests in tests/delta_stepping
// already run every strategy over a range of deltas, so one representative delta is enough here and
// keeps the number of collective single-source runs in check. testStandardVaryingDelta covers that the
// delta reaches the single-source computation at all.

namespace {
constexpr auto default_delta = distance_type{ 4 };

/**
 * @brief The expected outcome of an all-pairs shortest path computation on one of the test graphs.
 */
struct ApspExpectation {
    double average_shortest_path_length{};
    double average_efficiency{};
    distance_type diameter{};
    global_node_id_type number_disconnected_pairs{};
    /** The expected cluster sizes as (number of reachable nodes, number of nodes that reach that many) pairs */
    std::vector<std::pair<global_node_id_type, global_node_id_type>> cluster_sizes{};
};

/**
 * @brief Checks an all-pairs result against an expectation. Reports every deviating value instead of
 *		stopping at the first one; use it as the last statement of a test.
 * @param result The result of the computation
 * @param expected The expected values
 */
void expect_apsp(const ApspGlobalResult<distance_type>& result, const ApspExpectation& expected) {
    EXPECT_NEAR(result.average_shortest_path_length, expected.average_shortest_path_length, 1e-6) << "the average shortest path length";
    EXPECT_NEAR(result.average_efficiency, expected.average_efficiency, 1e-6) << "the average efficiency";
    EXPECT_EQ(result.diameter, expected.diameter) << "the diameter";
    EXPECT_EQ(result.number_disconnected_pairs, expected.number_disconnected_pairs) << "the number of disconnected pairs";

    ASSERT_EQ(result.cluster_sizes.size(), expected.cluster_sizes.size()) << "the number of distinct cluster sizes";

    for (const auto& [number_reachable_nodes, number_nodes] : expected.cluster_sizes) {
        ASSERT_TRUE(result.cluster_sizes.contains(number_reachable_nodes)) << "no node reaches " << number_reachable_nodes << " nodes";
        EXPECT_EQ(result.cluster_sizes.at(number_reachable_nodes), number_nodes) << "the number of nodes that reach " << number_reachable_nodes << " nodes";
    }
}

const auto standard_one_rank = ApspExpectation{ 3.1, 0.4222486772486772, 7, 0, { { 10, 10 } } };
const auto standard_uu_one_rank = ApspExpectation{ 1.6, 0.7222222222222222, 3, 0, { { 10, 10 } } };
const auto full_one_rank = ApspExpectation{ 1.0, 1.0, 1, 0, { { 10, 10 } } };

const auto standard_four_ranks = ApspExpectation{ 19.784615384615385, 0.13387777556694200, 40, 0, { { 40, 40 } } };
const auto standard_uu_four_ranks = ApspExpectation{ 2.5025641025641026, 0.48384615384615381, 5, 0, { { 40, 40 } } };
const auto full_four_ranks = ApspExpectation{ 1.0, 1.0, 1, 0, { { 36, 36 } } };

const auto standard_seven_ranks = ApspExpectation{ 17.587448559670783, 0.15517143563687033, 40, 2886, { { 1, 6 }, { 2, 4 }, { 20, 20 }, { 40, 40 } } };
const auto standard_uu_seven_ranks = ApspExpectation{ 2.4061538461538463, 0.50605128205128700, 5, 2880, { { 1, 3 }, { 2, 4 }, { 3, 3 }, { 20, 20 }, { 40, 40 } } };
const auto full_seven_ranks = ApspExpectation{ 1.0, 1.0, 1, 0, { { 63, 63 } } };

/**
 * @brief Names the parameterized tests after their epoch strategy instead of after its index.
 * @param info The parameter of the test
 * @return The name of the epoch strategy
 */
[[nodiscard]] std::string get_epoch_type_name(const testing::TestParamInfo<DeltaSteppingEpochType>& info) {
    switch (info.param) {
    case DeltaSteppingEpochType::PushBruteForce:
        return "PushBruteForce";
    case DeltaSteppingEpochType::PushCheck:
        return "PushCheck";
    case DeltaSteppingEpochType::PushShortLong:
        return "PushShortLong";
    case DeltaSteppingEpochType::PushInnerShort:
        return "PushInnerShort";
    case DeltaSteppingEpochType::PullModel:
        return "PullModel";
    }

    return "Unknown";
}
} // namespace

INSTANTIATE_TEST_SUITE_P(ApspDeltaTests, APSPDeltaSteppingTest,
                         testing::Values(DeltaSteppingEpochType::PushBruteForce, DeltaSteppingEpochType::PushCheck, DeltaSteppingEpochType::PushShortLong,
                                         DeltaSteppingEpochType::PushInnerShort, DeltaSteppingEpochType::PullModel),
                         get_epoch_type_name);

TEST_P(APSPDeltaSteppingTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_one_rank);
}

TEST_P(APSPDeltaSteppingTest, testStandardVaryingDelta) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    // One delta below every arc weight, one in the middle, and one above the diameter, i.e., one bucket
    // per arc weight, a few arcs per bucket, and a single bucket that holds the whole graph
    for (const auto current_delta : { distance_type{ 1 }, distance_type{ 3 }, distance_type{ 10000 } }) {
        const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, current_delta, GetParam());

        expect_apsp(result, standard_one_rank);
    }
}

TEST_P(APSPDeltaSteppingTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_uu_one_rank);
}

TEST_P(APSPDeltaSteppingTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, full_one_rank);
}

// The graphs of the four and seven MPI ranks are checked twice: once flattened onto a single rank, where
// the single-source runs need no communication at all, and once distributed, where every rank takes part
// in every one of them. Both have to report the same values.

TEST_P(APSPDeltaSteppingTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_four_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_uu_four_ranks);
}

TEST_P(APSPDeltaSteppingTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, full_four_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    // The single-source results are reduced onto the root rank, so only its running sums are the ones of the graph
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    expect_apsp(result, standard_four_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    expect_apsp(result, standard_uu_four_ranks);
}

TEST_P(APSPDeltaSteppingTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    expect_apsp(result, full_four_ranks);
}

// The seven-rank graphs are the ones with disconnected pairs: their ranks 2 and 5 form a component of
// their own and rank 3 holds isolated nodes, sinks, and sources, so a source that reaches nothing has to
// leave the running sums alone and still show up in the cluster sizes.

TEST_P(APSPDeltaSteppingTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_seven_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, standard_uu_seven_ranks);
}

TEST_P(APSPDeltaSteppingTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    expect_apsp(result, full_seven_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    expect_apsp(result, standard_seven_ranks);
}

TEST_P(APSPDeltaSteppingTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    expect_apsp(result, standard_uu_seven_ranks);
}

TEST_P(APSPDeltaSteppingTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto result = AllPairsShortestPath::compute_apsp_delta_stepping(graph, default_delta, GetParam());

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    expect_apsp(result, full_seven_ranks);
}
