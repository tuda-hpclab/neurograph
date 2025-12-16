/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_weight_counter.h"

#include "Types.h"

#include "metrics/WeightCounter.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>
#include <vector>

TEST_F(InWeightCounterTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, 45);

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 45);
}

TEST_F(InWeightCounterTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, 100);

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 100);
}

TEST_F(OutWeightCounterTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, 45);

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 45);
}

TEST_F(OutWeightCounterTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, 100);

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 100);
}

TEST_F(InWeightCounterTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_in_arcs = InWeightCounter::weigh_in_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_in_arcs, 608);
    }

    const auto total_number_of_in_arcs_global = InWeightCounter::all_weigh_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, 608);
}

TEST_F(InWeightCounterTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto total_number_of_out_arcs = OutWeightCounter::weigh_out_arcs(graph);
    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(total_number_of_out_arcs, 608);
    }

    const auto total_number_of_out_arcs_global = OutWeightCounter::all_weigh_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, 608);
}

TEST_F(OutWeightCounterTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
