/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_all_pairs_distances_histogram.h"

#include "metrics/AllPairsDistancesHistogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(AllPairsDistancesHistogramTest, testWidthStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 2.3);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 6);
    ASSERT_EQ(borders.size(), 6);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 7);
    ASSERT_EQ(counts[2], 7);
    ASSERT_EQ(counts[3], 19);
    ASSERT_EQ(counts[4], 7);
    ASSERT_EQ(counts[5], 3);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 2.3, 1e-6);
    ASSERT_NEAR(borders[2], 4.6, 1e-6);
    ASSERT_NEAR(borders[3], 6.9, 1e-6);
    ASSERT_NEAR(borders[4], 9.2, 1e-6);
    ASSERT_NEAR(borders[5], 11.5, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 1.7);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 8);
    ASSERT_EQ(borders.size(), 8);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 3);
    ASSERT_EQ(counts[2], 4);
    ASSERT_EQ(counts[3], 7);
    ASSERT_EQ(counts[4], 14);
    ASSERT_EQ(counts[5], 8);
    ASSERT_EQ(counts[6], 6);
    ASSERT_EQ(counts[7], 1);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 1.7, 1e-6);
    ASSERT_NEAR(borders[2], 3.4, 1e-6);
    ASSERT_NEAR(borders[3], 5.1, 1e-6);
    ASSERT_NEAR(borders[4], 6.8, 1e-6);
    ASSERT_NEAR(borders[5], 8.5, 1e-6);
    ASSERT_NEAR(borders[6], 10.2, 1e-6);
    ASSERT_NEAR(borders[7], 11.9, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 3.8);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 8);
    ASSERT_EQ(counts[1], 13);
    ASSERT_EQ(counts[2], 21);
    ASSERT_EQ(counts[3], 3);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 3.8, 1e-6);
    ASSERT_NEAR(borders[2], 7.6, 1e-6);
    ASSERT_NEAR(borders[3], 11.4, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 2.3);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 6);
    ASSERT_EQ(borders.size(), 6);

    ASSERT_EQ(counts[0], 437);
    ASSERT_EQ(counts[1], 18);
    ASSERT_EQ(counts[2], 33);
    ASSERT_EQ(counts[3], 111);
    ASSERT_EQ(counts[4], 141);
    ASSERT_EQ(counts[5], 40);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 2.3, 1e-6);
    ASSERT_NEAR(borders[2], 4.6, 1e-6);
    ASSERT_NEAR(borders[3], 6.9, 1e-6);
    ASSERT_NEAR(borders[4], 9.2, 1e-6);
    ASSERT_NEAR(borders[5], 11.5, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 1.7);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 8);
    ASSERT_EQ(borders.size(), 8);

    ASSERT_EQ(counts[0], 437);
    ASSERT_EQ(counts[1], 3);
    ASSERT_EQ(counts[2], 29);
    ASSERT_EQ(counts[3], 15);
    ASSERT_EQ(counts[4], 80);
    ASSERT_EQ(counts[5], 80);
    ASSERT_EQ(counts[6], 113);
    ASSERT_EQ(counts[7], 23);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 1.7, 1e-6);
    ASSERT_NEAR(borders[2], 3.4, 1e-6);
    ASSERT_NEAR(borders[3], 5.1, 1e-6);
    ASSERT_NEAR(borders[4], 6.8, 1e-6);
    ASSERT_NEAR(borders[5], 8.5, 1e-6);
    ASSERT_NEAR(borders[6], 10.2, 1e-6);
    ASSERT_NEAR(borders[7], 11.9, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 3.8);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 382);
    ASSERT_EQ(counts[1], 61);
    ASSERT_EQ(counts[2], 142);
    ASSERT_EQ(counts[3], 45);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 3.8, 1e-6);
    ASSERT_NEAR(borders[2], 7.6, 1e-6);
    ASSERT_NEAR(borders[3], 11.4, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 2.3);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 6);
    ASSERT_EQ(borders.size(), 6);

    ASSERT_EQ(counts[0], 437);
    ASSERT_EQ(counts[1], 18);
    ASSERT_EQ(counts[2], 33);
    ASSERT_EQ(counts[3], 111);
    ASSERT_EQ(counts[4], 141);
    ASSERT_EQ(counts[5], 40);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 2.3, 1e-6);
    ASSERT_NEAR(borders[2], 4.6, 1e-6);
    ASSERT_NEAR(borders[3], 6.9, 1e-6);
    ASSERT_NEAR(borders[4], 9.2, 1e-6);
    ASSERT_NEAR(borders[5], 11.5, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 1.7);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 8);
    ASSERT_EQ(borders.size(), 8);

    ASSERT_EQ(counts[0], 437);
    ASSERT_EQ(counts[1], 3);
    ASSERT_EQ(counts[2], 29);
    ASSERT_EQ(counts[3], 15);
    ASSERT_EQ(counts[4], 80);
    ASSERT_EQ(counts[5], 80);
    ASSERT_EQ(counts[6], 113);
    ASSERT_EQ(counts[7], 23);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 1.7, 1e-6);
    ASSERT_NEAR(borders[2], 3.4, 1e-6);
    ASSERT_NEAR(borders[3], 5.1, 1e-6);
    ASSERT_NEAR(borders[4], 6.8, 1e-6);
    ASSERT_NEAR(borders[5], 8.5, 1e-6);
    ASSERT_NEAR(borders[6], 10.2, 1e-6);
    ASSERT_NEAR(borders[7], 11.9, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testWidthFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 3.8);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 382);
    ASSERT_EQ(counts[1], 61);
    ASSERT_EQ(counts[2], 142);
    ASSERT_EQ(counts[3], 45);

    ASSERT_NEAR(borders[0], 0, 1e-6);
    ASSERT_NEAR(borders[1], 3.8, 1e-6);
    ASSERT_NEAR(borders[2], 7.6, 1e-6);
    ASSERT_NEAR(borders[3], 11.4, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 12.0, 4);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 10);
    ASSERT_EQ(counts[2], 21);
    ASSERT_EQ(counts[3], 12);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 6.0, 1e-6);
    ASSERT_NEAR(borders[3], 9.0, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 13.5, 2);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 2);
    ASSERT_EQ(borders.size(), 2);

    ASSERT_EQ(counts[0], 15);
    ASSERT_EQ(counts[1], 30);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 6.75, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 12.3, 7);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // These values are calculated manually
    ASSERT_EQ(counts.size(), 7);
    ASSERT_EQ(borders.size(), 7);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 4);
    ASSERT_EQ(counts[2], 3);
    ASSERT_EQ(counts[3], 7);
    ASSERT_EQ(counts[4], 15);
    ASSERT_EQ(counts[5], 7);
    ASSERT_EQ(counts[6], 7);

    ASSERT_NEAR(borders[0], 12.3 / 7 * 0, 1e-6);
    ASSERT_NEAR(borders[1], 12.3 / 7 * 1, 1e-6);
    ASSERT_NEAR(borders[2], 12.3 / 7 * 2, 1e-6);
    ASSERT_NEAR(borders[3], 12.3 / 7 * 3, 1e-6);
    ASSERT_NEAR(borders[4], 12.3 / 7 * 4, 1e-6);
    ASSERT_NEAR(borders[5], 12.3 / 7 * 5, 1e-6);
    ASSERT_NEAR(borders[6], 12.3 / 7 * 6, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 16, 4);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 443);
    ASSERT_EQ(counts[1], 88);
    ASSERT_EQ(counts[2], 238);
    ASSERT_EQ(counts[3], 11);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 4.0, 1e-6);
    ASSERT_NEAR(borders[2], 8.0, 1e-6);
    ASSERT_NEAR(borders[3], 12.0, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 13.5, 3);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 3);
    ASSERT_EQ(borders.size(), 3);

    ASSERT_EQ(counts[0], 451);
    ASSERT_EQ(counts[1], 137);
    ASSERT_EQ(counts[2], 192);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 4.5, 1e-6);
    ASSERT_NEAR(borders[2], 9.0, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 13.8, 7);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(counts.size(), 7);
    ASSERT_EQ(borders.size(), 7);

    ASSERT_EQ(counts[0], 380);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 30);
    ASSERT_EQ(counts[3], 35);
    ASSERT_EQ(counts[4], 62);
    ASSERT_EQ(counts[5], 93);
    ASSERT_EQ(counts[6], 28);

    ASSERT_NEAR(borders[0], 13.8 / 7 * 0, 1e-6);
    ASSERT_NEAR(borders[1], 13.8 / 7 * 1, 1e-6);
    ASSERT_NEAR(borders[2], 13.8 / 7 * 2, 1e-6);
    ASSERT_NEAR(borders[3], 13.8 / 7 * 3, 1e-6);
    ASSERT_NEAR(borders[4], 13.8 / 7 * 4, 1e-6);
    ASSERT_NEAR(borders[5], 13.8 / 7 * 5, 1e-6);
    ASSERT_NEAR(borders[6], 13.8 / 7 * 6, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 16, 4);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 4);
    ASSERT_EQ(borders.size(), 4);

    ASSERT_EQ(counts[0], 443);
    ASSERT_EQ(counts[1], 88);
    ASSERT_EQ(counts[2], 238);
    ASSERT_EQ(counts[3], 11);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 4.0, 1e-6);
    ASSERT_NEAR(borders[2], 8.0, 1e-6);
    ASSERT_NEAR(borders[3], 12.0, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 13.5, 3);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 3);
    ASSERT_EQ(borders.size(), 3);

    ASSERT_EQ(counts[0], 451);
    ASSERT_EQ(counts[1], 137);
    ASSERT_EQ(counts[2], 192);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 4.5, 1e-6);
    ASSERT_NEAR(borders[2], 9.0, 1e-6);
}

TEST_F(AllPairsDistancesHistogramTest, testCountFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 13.8, 7);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(counts.size(), 7);
    ASSERT_EQ(borders.size(), 7);

    ASSERT_EQ(counts[0], 380);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 30);
    ASSERT_EQ(counts[3], 35);
    ASSERT_EQ(counts[4], 62);
    ASSERT_EQ(counts[5], 93);
    ASSERT_EQ(counts[6], 28);

    ASSERT_NEAR(borders[0], 13.8 / 7 * 0, 1e-6);
    ASSERT_NEAR(borders[1], 13.8 / 7 * 1, 1e-6);
    ASSERT_NEAR(borders[2], 13.8 / 7 * 2, 1e-6);
    ASSERT_NEAR(borders[3], 13.8 / 7 * 3, 1e-6);
    ASSERT_NEAR(borders[4], 13.8 / 7 * 4, 1e-6);
    ASSERT_NEAR(borders[5], 13.8 / 7 * 5, 1e-6);
    ASSERT_NEAR(borders[6], 13.8 / 7 * 6, 1e-6);
}
