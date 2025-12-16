/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_length.h"

#include "metrics/ArcLength.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(ArcLengthTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 5.646332800503403, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 10.860479, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 5.646332800503403, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 10.860479, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 6.472791682288855, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 10.860479, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 6.472791682288855, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 10.860479, 1e-6);
}

TEST_F(ArcLengthTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_arc_length, 6.662768699459406, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 11.969126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 6.662768699459406, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 11.969126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 4.0768878773841095, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 4.0768878773841095, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(maximum_arc_length, 12.544612, 1e-6);

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    ASSERT_NEAR(all_average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.544612, 1e-6);
}

TEST_F(ArcLengthTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 4.0768878773841095, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 4.0768878773841095, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 2.998320428459754, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.0, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.431126, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 2.998320428459754, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.0, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.431126, 1e-6);
}

TEST_F(ArcLengthTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_arc_length = ArcLength::compute_average_arc_length(graph);
    const auto minimum_arc_length = ArcLength::compute_minimum_arc_length(graph);
    const auto maximum_arc_length = ArcLength::compute_maximum_arc_length(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_arc_length, 4.0564215744513206, 1e-6);
        ASSERT_NEAR(minimum_arc_length, 0.042426, 1e-6);
        ASSERT_NEAR(maximum_arc_length, 12.544612, 1e-6);
    }

    const auto all_average_arc_length = ArcLength::all_compute_average_arc_length(graph);
    const auto all_minimum_arc_length = ArcLength::all_compute_minimum_arc_length(graph);
    const auto all_maximum_arc_length = ArcLength::all_compute_maximum_arc_length(graph);

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(all_average_arc_length, 4.0564215744513206, 1e-6);
    ASSERT_NEAR(all_minimum_arc_length, 0.042426, 1e-6);
    ASSERT_NEAR(all_maximum_arc_length, 12.544612, 1e-6);
}
