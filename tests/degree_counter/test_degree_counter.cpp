/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree_counter.h"

#include "metrics/DegreeCounter.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(AverageDegreeTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_in_in, 4.0, 1e-6);
    ASSERT_NEAR(average_in_out, 6.06666666666666, 1e-6);
    ASSERT_NEAR(average_out_in, 8.5111111111111111, 1e-6);
    ASSERT_NEAR(average_out_out, 4.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_in_in, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_in_out, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_in, 4.590909090909090909090, 1e-6);
    ASSERT_NEAR(average_out_out, 4.590909090909090909090, 1e-6);
}

TEST_F(AverageDegreeTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_in_in, 10.0, 1e-6);
    ASSERT_NEAR(average_in_out, 10.0, 1e-6);
    ASSERT_NEAR(average_out_in, 10.0, 1e-6);
    ASSERT_NEAR(average_out_out, 10.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_in_in, 15.085526315789474, 1e-6);
    ASSERT_NEAR(average_in_out, 15.736842105263158, 1e-6);
    ASSERT_NEAR(average_out_in, 15.947368421052632, 1e-6);
    ASSERT_NEAR(average_out_out, 15.085526315789474, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_in_in, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out, 6.53125, 1e-6);
}

TEST_F(AverageDegreeTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_in_in, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out, 35.0, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in, 15.085526315789474, 1e-6);
    ASSERT_NEAR(average_in_out, 15.736842105263158, 1e-6);
    ASSERT_NEAR(average_out_in, 15.947368421052632, 1e-6);
    ASSERT_NEAR(average_out_out, 15.085526315789474, 1e-6);
}

TEST_F(AverageDegreeTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in, 6.53125, 1e-6);
    ASSERT_NEAR(average_in_out, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_in, 6.53125, 1e-6);
    ASSERT_NEAR(average_out_out, 6.53125, 1e-6);
}

TEST_F(AverageDegreeTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_in_in = AverageDegree::all_compute_average_in_in_degree(graph);
    const auto average_in_out = AverageDegree::all_compute_average_in_out_degree(graph);
    const auto average_out_in = AverageDegree::all_compute_average_out_in_degree(graph);
    const auto average_out_out = AverageDegree::all_compute_average_out_out_degree(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_in_in, 35.0, 1e-6);
    ASSERT_NEAR(average_in_out, 35.0, 1e-6);
    ASSERT_NEAR(average_out_in, 35.0, 1e-6);
    ASSERT_NEAR(average_out_out, 35.0, 1e-6);
}
