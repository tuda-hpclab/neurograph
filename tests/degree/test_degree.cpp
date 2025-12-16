/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree.h"

#include "metrics/Degree.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(DegreeTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 1);
    ASSERT_EQ(maximum_in_1, 5);
    ASSERT_EQ(minimum_in_2, 1);
    ASSERT_EQ(maximum_in_2, 5);

    ASSERT_EQ(all_minimum_in_1, 1);
    ASSERT_EQ(all_maximum_in_1, 5);
    ASSERT_EQ(all_minimum_in_2, 1);
    ASSERT_EQ(all_maximum_in_2, 5);

    ASSERT_EQ(minimum_out_1, 2);
    ASSERT_EQ(maximum_out_1, 4);
    ASSERT_EQ(minimum_out_2, 2);
    ASSERT_EQ(maximum_out_2, 4);

    ASSERT_EQ(all_minimum_out_1, 2);
    ASSERT_EQ(all_maximum_out_1, 4);
    ASSERT_EQ(all_minimum_out_2, 2);
    ASSERT_EQ(all_maximum_out_2, 4);
}

TEST_F(DegreeTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 3);
    ASSERT_EQ(maximum_in_1, 6);
    ASSERT_EQ(minimum_in_2, 3);
    ASSERT_EQ(maximum_in_2, 6);

    ASSERT_EQ(all_minimum_in_1, 3);
    ASSERT_EQ(all_maximum_in_1, 6);
    ASSERT_EQ(all_minimum_in_2, 3);
    ASSERT_EQ(all_maximum_in_2, 6);

    ASSERT_EQ(minimum_out_1, 3);
    ASSERT_EQ(maximum_out_1, 6);
    ASSERT_EQ(minimum_out_2, 3);
    ASSERT_EQ(maximum_out_2, 6);

    ASSERT_EQ(all_minimum_out_1, 3);
    ASSERT_EQ(all_maximum_out_1, 6);
    ASSERT_EQ(all_minimum_out_2, 3);
    ASSERT_EQ(all_maximum_out_2, 6);
}

TEST_F(DegreeTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values come from manually counting
    ASSERT_EQ(minimum_in_1, 10);
    ASSERT_EQ(maximum_in_1, 10);
    ASSERT_EQ(minimum_in_2, 10);
    ASSERT_EQ(maximum_in_2, 10);

    ASSERT_EQ(all_minimum_in_1, 10);
    ASSERT_EQ(all_maximum_in_1, 10);
    ASSERT_EQ(all_minimum_in_2, 10);
    ASSERT_EQ(all_maximum_in_2, 10);

    ASSERT_EQ(minimum_out_1, 10);
    ASSERT_EQ(maximum_out_1, 10);
    ASSERT_EQ(minimum_out_2, 10);
    ASSERT_EQ(maximum_out_2, 10);

    ASSERT_EQ(all_minimum_out_1, 10);
    ASSERT_EQ(all_maximum_out_1, 10);
    ASSERT_EQ(all_minimum_out_2, 10);
    ASSERT_EQ(all_maximum_out_2, 10);
}

TEST_F(DegreeTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 2);
    ASSERT_EQ(maximum_in_1, 5);
    ASSERT_EQ(minimum_in_2, 2);
    ASSERT_EQ(maximum_in_2, 5);

    ASSERT_EQ(all_minimum_in_1, 2);
    ASSERT_EQ(all_maximum_in_1, 5);
    ASSERT_EQ(all_minimum_in_2, 2);
    ASSERT_EQ(all_maximum_in_2, 5);

    ASSERT_EQ(minimum_out_1, 2);
    ASSERT_EQ(maximum_out_1, 5);
    ASSERT_EQ(minimum_out_2, 2);
    ASSERT_EQ(maximum_out_2, 5);

    ASSERT_EQ(all_minimum_out_1, 2);
    ASSERT_EQ(all_maximum_out_1, 5);
    ASSERT_EQ(all_minimum_out_2, 2);
    ASSERT_EQ(all_maximum_out_2, 5);
}

TEST_F(DegreeTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 5);
    ASSERT_EQ(maximum_in_1, 8);
    ASSERT_EQ(minimum_in_2, 5);
    ASSERT_EQ(maximum_in_2, 8);

    ASSERT_EQ(all_minimum_in_1, 5);
    ASSERT_EQ(all_maximum_in_1, 8);
    ASSERT_EQ(all_minimum_in_2, 5);
    ASSERT_EQ(all_maximum_in_2, 8);

    ASSERT_EQ(minimum_out_1, 5);
    ASSERT_EQ(maximum_out_1, 8);
    ASSERT_EQ(minimum_out_2, 5);
    ASSERT_EQ(maximum_out_2, 8);

    ASSERT_EQ(all_minimum_out_1, 5);
    ASSERT_EQ(all_maximum_out_1, 8);
    ASSERT_EQ(all_minimum_out_2, 5);
    ASSERT_EQ(all_maximum_out_2, 8);
}

TEST_F(DegreeTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(minimum_in_1, 35);
    ASSERT_EQ(maximum_in_1, 35);
    ASSERT_EQ(minimum_in_2, 35);
    ASSERT_EQ(maximum_in_2, 35);

    ASSERT_EQ(all_minimum_in_1, 35);
    ASSERT_EQ(all_maximum_in_1, 35);
    ASSERT_EQ(all_minimum_in_2, 35);
    ASSERT_EQ(all_maximum_in_2, 35);

    ASSERT_EQ(minimum_out_1, 35);
    ASSERT_EQ(maximum_out_1, 35);
    ASSERT_EQ(minimum_out_2, 35);
    ASSERT_EQ(maximum_out_2, 35);

    ASSERT_EQ(all_minimum_out_1, 35);
    ASSERT_EQ(all_maximum_out_1, 35);
    ASSERT_EQ(all_minimum_out_2, 35);
    ASSERT_EQ(all_maximum_out_2, 35);
}

TEST_F(DegreeTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 2);
        ASSERT_EQ(maximum_in_1, 5);
        ASSERT_EQ(minimum_in_2, 2);
        ASSERT_EQ(maximum_in_2, 5);
    }

    ASSERT_EQ(all_minimum_in_1, 2);
    ASSERT_EQ(all_maximum_in_1, 5);
    ASSERT_EQ(all_minimum_in_2, 2);
    ASSERT_EQ(all_maximum_in_2, 5);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 2);
        ASSERT_EQ(maximum_out_1, 5);
        ASSERT_EQ(minimum_out_2, 2);
        ASSERT_EQ(maximum_out_2, 5);
    }

    ASSERT_EQ(all_minimum_out_1, 2);
    ASSERT_EQ(all_maximum_out_1, 5);
    ASSERT_EQ(all_minimum_out_2, 2);
    ASSERT_EQ(all_maximum_out_2, 5);
}

TEST_F(DegreeTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 5);
        ASSERT_EQ(maximum_in_1, 8);
        ASSERT_EQ(minimum_in_2, 5);
        ASSERT_EQ(maximum_in_2, 8);
    }

    ASSERT_EQ(all_minimum_in_1, 5);
    ASSERT_EQ(all_maximum_in_1, 8);
    ASSERT_EQ(all_minimum_in_2, 5);
    ASSERT_EQ(all_maximum_in_2, 8);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 5);
        ASSERT_EQ(maximum_out_1, 8);
        ASSERT_EQ(minimum_out_2, 5);
        ASSERT_EQ(maximum_out_2, 8);
    }

    ASSERT_EQ(all_minimum_out_1, 5);
    ASSERT_EQ(all_maximum_out_1, 8);
    ASSERT_EQ(all_minimum_out_2, 5);
    ASSERT_EQ(all_maximum_out_2, 8);
}

TEST_F(DegreeTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto minimum_in_1 = Degree::compute_minimum_in_degree(graph);
    const auto maximum_in_1 = Degree::compute_maximum_in_degree(graph);
    const auto& [minimum_in_2, maximum_in_2] = Degree::compute_extreme_in_degrees(graph);

    const auto all_minimum_in_1 = Degree::all_compute_minimum_in_degree(graph);
    const auto all_maximum_in_1 = Degree::all_compute_maximum_in_degree(graph);
    const auto& [all_minimum_in_2, all_maximum_in_2] = Degree::all_compute_extreme_in_degrees(graph);

    const auto minimum_out_1 = Degree::compute_minimum_out_degree(graph);
    const auto maximum_out_1 = Degree::compute_maximum_out_degree(graph);
    const auto& [minimum_out_2, maximum_out_2] = Degree::compute_extreme_out_degrees(graph);

    const auto all_minimum_out_1 = Degree::all_compute_minimum_out_degree(graph);
    const auto all_maximum_out_1 = Degree::all_compute_maximum_out_degree(graph);
    const auto& [all_minimum_out_2, all_maximum_out_2] = Degree::all_compute_extreme_out_degrees(graph);

    // The values are the corresponding ones from the one-rank version

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_in_1, 35);
        ASSERT_EQ(maximum_in_1, 35);
        ASSERT_EQ(minimum_in_2, 35);
        ASSERT_EQ(maximum_in_2, 35);
    }

    ASSERT_EQ(all_minimum_in_1, 35);
    ASSERT_EQ(all_maximum_in_1, 35);
    ASSERT_EQ(all_minimum_in_2, 35);
    ASSERT_EQ(all_maximum_in_2, 35);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(minimum_out_1, 35);
        ASSERT_EQ(maximum_out_1, 35);
        ASSERT_EQ(minimum_out_2, 35);
        ASSERT_EQ(maximum_out_2, 35);
    }

    ASSERT_EQ(all_minimum_out_1, 35);
    ASSERT_EQ(all_maximum_out_1, 35);
    ASSERT_EQ(all_minimum_out_2, 35);
    ASSERT_EQ(all_maximum_out_2, 35);
}
