/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_all_pairs_distances.h"

#include "metrics/AllPairsDistances.h"

#include "mpi-wrapper/MPIInfo.h"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(AllPairsDistancesTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(maximum_distance, 11.969126, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values comes from manual calculations
    ASSERT_NEAR(all_average_distance, 7.403076, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 1.170469, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 11.969126, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(all_average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.396961, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.396961, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.396961, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}

TEST_F(AllPairsDistancesTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_distance = AllPairsDistances::compute_average_pair_distance(graph);
    const auto minimum_distance = AllPairsDistances::compute_minimum_pair_distance(graph);
    const auto maximum_distance = AllPairsDistances::compute_maximum_pair_distance(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_distance, 4.056421, 1e-6);
        ASSERT_NEAR(minimum_distance, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_distance, 12.544612, 1e-6);
    }

    const auto all_average_distance = AllPairsDistances::all_compute_average_pair_distance(graph);
    const auto all_minimum_distance = AllPairsDistances::all_compute_minimum_pair_distance(graph);
    const auto all_maximum_distance = AllPairsDistances::all_compute_maximum_pair_distance(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_distance, 4.056421, 1e-6);
    ASSERT_NEAR(all_minimum_distance, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_distance, 12.544612, 1e-6);
}
