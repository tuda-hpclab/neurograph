/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_node_counter.h"

#include "Types.h"

#include "metrics/counting/NodeCounter.h"
#include "metrics/counting/NodeDistributionCounter.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>
#include <vector>

TEST_F(NodeCounterTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    ASSERT_EQ(total_number_nodes, graph.get_number_local_nodes());

    const auto total_number_nodes_global = NodeCounter::all_count_nodes(graph);
    ASSERT_EQ(total_number_nodes_global, graph.get_number_local_nodes());
}

TEST_F(NodeCounterTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    ASSERT_EQ(total_number_nodes, graph.get_number_local_nodes());

    const auto total_number_nodes_global = NodeCounter::all_count_nodes(graph);
    ASSERT_EQ(total_number_nodes_global, graph.get_number_local_nodes());
}

TEST_F(NodeCounterTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_nodes, 40);
    }

    const auto total_number_nodes_global = NodeCounter::all_count_nodes(graph);
    ASSERT_EQ(total_number_nodes_global, 40);
}

TEST_F(NodeCounterTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_nodes, 36);
    }

    const auto total_number_nodes_global = NodeCounter::all_count_nodes(graph);
    ASSERT_EQ(total_number_nodes_global, 36);
}

TEST_F(NodeDistributionCounterTest, testHistogramStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    ASSERT_EQ(histogram, std::vector<node_id_type>{ 10 });

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>{ 10 });
}

TEST_F(NodeDistributionCounterTest, testHistogramFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    ASSERT_EQ(histogram, std::vector<node_id_type>{ 10 });

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>{ 10 });
}

TEST_F(NodeDistributionCounterTest, testHistogramStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(histogram, std::vector<node_id_type>({ 10, 10, 10, 10 }));
    }

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>({ 10, 10, 10, 10 }));
}

TEST_F(NodeDistributionCounterTest, testHistogramFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(histogram, std::vector<node_id_type>({ 8, 9, 9, 10 }));
    }

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>({ 8, 9, 9, 10 }));
}

TEST_F(NodeCounterTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // Seven blocks of ten nodes each. The value serves as standard for the seven-rank test
    ASSERT_EQ(NodeCounter::count_nodes(graph), 70);
    ASSERT_EQ(NodeCounter::all_count_nodes(graph), 70);
}

TEST_F(NodeCounterTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // 8 + 9 + 8 + 10 + 9 + 9 + 10 nodes. The value serves as standard for the seven-rank test
    ASSERT_EQ(NodeCounter::count_nodes(graph), 63);
    ASSERT_EQ(NodeCounter::all_count_nodes(graph), 63);
}

TEST_F(NodeCounterTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_nodes, 70);
    }

    ASSERT_EQ(NodeCounter::all_count_nodes(graph), 70);
}

TEST_F(NodeCounterTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto total_number_nodes = NodeCounter::count_nodes(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_nodes, 63);
    }

    ASSERT_EQ(NodeCounter::all_count_nodes(graph), 63);
}

TEST_F(NodeDistributionCounterTest, testHistogramStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(histogram, std::vector<node_id_type>({ 10, 10, 10, 10, 10, 10, 10 }));
    }

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>({ 10, 10, 10, 10, 10, 10, 10 }));
}

TEST_F(NodeDistributionCounterTest, testHistogramFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto histogram = NodeDistributionCounter::all_count_node_distribution(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(histogram, std::vector<node_id_type>({ 8, 9, 8, 10, 9, 9, 10 }));
    }

    const auto histogram_global = NodeDistributionCounter::all_count_node_distribution_global(graph);
    ASSERT_EQ(histogram_global, std::vector<global_node_id_type>({ 8, 9, 8, 10, 9, 9, 10 }));
}
