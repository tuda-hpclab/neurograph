/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_counter.h"

#include "metrics/ArcCounter.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(InArcCounterTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, graph.get_number_local_in_arcs());

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, graph.get_number_local_in_arcs());
}

TEST_F(InArcCounterTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_in_arcs = InArcCounter::count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs, graph.get_number_local_in_arcs());

    const auto total_number_of_in_arcs_global = InArcCounter::all_count_in_arcs(graph);
    ASSERT_EQ(total_number_of_in_arcs_global, graph.get_number_local_in_arcs());
}

TEST_F(OutArcCounterTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, graph.get_number_local_out_arcs());

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, graph.get_number_local_out_arcs());
}

TEST_F(OutArcCounterTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto total_number_of_out_arcs = OutArcCounter::count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs, graph.get_number_local_out_arcs());

    const auto total_number_of_out_arcs_global = OutArcCounter::all_count_out_arcs(graph);
    ASSERT_EQ(total_number_of_out_arcs_global, graph.get_number_local_out_arcs());
}

TEST_F(InArcCounterTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
