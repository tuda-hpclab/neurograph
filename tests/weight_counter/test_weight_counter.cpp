/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_weight_counter.h"

#include "Types.h"

#include "metrics/counting/InWeightCounter.h"
#include "metrics/counting/OutWeightCounter.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>
#include <vector>

TEST_F(InWeightCounterTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, 45);

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 45);
}

TEST_F(InWeightCounterTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, 100);

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 100);
}

TEST_F(OutWeightCounterTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, 45);

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 45);
}

TEST_F(OutWeightCounterTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, 100);

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 100);
}

TEST_F(InWeightCounterTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 620);
    }

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 620);
}

TEST_F(InWeightCounterTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 36 * 35);
    }

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 36 * 35);
}

TEST_F(OutWeightCounterTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 620);
    }

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 620);
}

TEST_F(OutWeightCounterTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 36 * 35);
    }

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 36 * 35);
}

TEST_F(InWeightCounterTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // Every standard block carries 45 units of intra-block weight plus 10 ring arcs of weight 11,
    // so six blocks contribute 6 * 155; the source/sink block adds 1 + 2 + 3 + 4.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(InWeightCounter::weigh_in_arcs(graph), 940);
    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 940);
}

TEST_F(InWeightCounterTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // Every arc has weight one, so the weight is the number of arcs.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(InWeightCounter::weigh_in_arcs(graph), 372);
    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 372);
}

TEST_F(InWeightCounterTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // Every one of the 63 * 62 arcs of the complete graph has weight one.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(InWeightCounter::weigh_in_arcs(graph), 3906);
    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 3906);
}

TEST_F(OutWeightCounterTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(OutWeightCounter::weigh_out_arcs(graph), 940);
    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 940);
}

TEST_F(OutWeightCounterTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(OutWeightCounter::weigh_out_arcs(graph), 372);
    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 372);
}

TEST_F(OutWeightCounterTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The value serves as standard for the seven-rank test
    ASSERT_EQ(OutWeightCounter::weigh_out_arcs(graph), 3906);
    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 3906);
}

TEST_F(InWeightCounterTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto total_weight = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 940);
    }

    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 940);
}

TEST_F(InWeightCounterTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto total_weight = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 372);
    }

    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 372);
}

TEST_F(InWeightCounterTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto total_weight = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 3906);
    }

    ASSERT_EQ(InWeightCounter::all_weigh_in_arcs(graph), 3906);
}

TEST_F(OutWeightCounterTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto total_weight = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 940);
    }

    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 940);
}

TEST_F(OutWeightCounterTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto total_weight = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 372);
    }

    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 372);
}

TEST_F(OutWeightCounterTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto total_weight = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_weight, 3906);
    }

    ASSERT_EQ(OutWeightCounter::all_weigh_out_arcs(graph), 3906);
}
