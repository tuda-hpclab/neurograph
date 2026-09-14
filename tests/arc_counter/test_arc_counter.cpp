/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_counter.h"

#include "metrics/counting/InArcCounter.h"
#include "metrics/counting/OutArcCounter.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(InArcCounterTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, graph.get_number_local_in_arcs());

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, graph.get_number_local_in_arcs());
}

TEST_F(InArcCounterTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, graph.get_number_local_in_arcs());

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, graph.get_number_local_in_arcs());
}

TEST_F(OutArcCounterTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, graph.get_number_local_out_arcs());

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, graph.get_number_local_out_arcs());
}

TEST_F(OutArcCounterTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, graph.get_number_local_out_arcs());

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, graph.get_number_local_out_arcs());
}

TEST_F(InArcCounterTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, graph.get_number_local_in_arcs() * 4);
    }

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, graph.get_number_local_in_arcs() * 4);
}

TEST_F(InArcCounterTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 36 * 35);
    }

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 36 * 35);
}

TEST_F(OutArcCounterTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, graph.get_number_local_out_arcs() * 4);
    }

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, graph.get_number_local_out_arcs() * 4);
}

TEST_F(OutArcCounterTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 36 * 35);
    }

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 36 * 35);
}

TEST_F(InArcCounterTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // Six blocks of 25 intra-block arcs plus 10 ring arcs each, and 4 arcs in the source/sink block.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(InArcCounter::count_in_arcs(graph), 214);
    ASSERT_EQ(InArcCounter::all_count_in_arcs(graph), 214);
}

TEST_F(InArcCounterTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // Every one of the 63 nodes is connected to every other one: 63 * 62 arcs.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(InArcCounter::count_in_arcs(graph), 3906);
    ASSERT_EQ(InArcCounter::all_count_in_arcs(graph), 3906);
}

TEST_F(OutArcCounterTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(OutArcCounter::count_out_arcs(graph), 214);
    ASSERT_EQ(OutArcCounter::all_count_out_arcs(graph), 214);
}

TEST_F(OutArcCounterTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(OutArcCounter::count_out_arcs(graph), 3906);
    ASSERT_EQ(OutArcCounter::all_count_out_arcs(graph), 3906);
}

TEST_F(InArcCounterTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 214);
    }

    ASSERT_EQ(InArcCounter::all_count_in_arcs(graph), 214);
}

TEST_F(InArcCounterTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 3906);
    }

    ASSERT_EQ(InArcCounter::all_count_in_arcs(graph), 3906);
}

TEST_F(OutArcCounterTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 214);
    }

    ASSERT_EQ(OutArcCounter::all_count_out_arcs(graph), 214);
}

TEST_F(OutArcCounterTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 3906);
    }

    ASSERT_EQ(OutArcCounter::all_count_out_arcs(graph), 3906);
}
