/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_length_histogram.h"

#include "metrics/ArcLengthHistogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(ArcLengthHistogramTest, testWidthStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 11);
    ASSERT_EQ(counts.size(), 11);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);

    ASSERT_EQ(counts[0], 3);
    ASSERT_EQ(counts[1], 3);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 14);
    ASSERT_EQ(counts[4], 1);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 1);
    ASSERT_EQ(counts[7], 7);
    ASSERT_EQ(counts[8], 7);
    ASSERT_EQ(counts[9], 6);
    ASSERT_EQ(counts[10], 1);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 11);
    ASSERT_EQ(counts.size(), 11);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 8);
    ASSERT_EQ(counts[4], 2);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 2);
    ASSERT_EQ(counts[7], 10);
    ASSERT_EQ(counts[8], 8);
    ASSERT_EQ(counts[9], 6);
    ASSERT_EQ(counts[10], 2);
}

TEST_F(ArcLengthHistogramTest, testWidthFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 12);
    ASSERT_EQ(counts.size(), 12);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);

    ASSERT_EQ(counts[0], 10);
    ASSERT_EQ(counts[1], 4);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 12);
    ASSERT_EQ(counts[4], 2);
    ASSERT_EQ(counts[5], 6);
    ASSERT_EQ(counts[6], 8);
    ASSERT_EQ(counts[7], 16);
    ASSERT_EQ(counts[8], 18);
    ASSERT_EQ(counts[9], 10);
    ASSERT_EQ(counts[10], 2);
    ASSERT_EQ(counts[11], 12);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 311);
    ASSERT_EQ(counts[1], 41);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 14);
    ASSERT_EQ(counts[4], 23);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 23);
    ASSERT_EQ(counts[7], 18);
    ASSERT_EQ(counts[8], 40);
    ASSERT_EQ(counts[9], 25);
    ASSERT_EQ(counts[10], 56);
    ASSERT_EQ(counts[11], 33);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 154);
    ASSERT_EQ(counts[1], 22);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 8);
    ASSERT_EQ(counts[4], 6);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 6);
    ASSERT_EQ(counts[7], 12);
    ASSERT_EQ(counts[8], 14);
    ASSERT_EQ(counts[9], 10);
    ASSERT_EQ(counts[10], 12);
    ASSERT_EQ(counts[11], 6);
    ASSERT_EQ(counts[12], 4);
}

TEST_F(ArcLengthHistogramTest, testWidthFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 650);
    ASSERT_EQ(counts[1], 110);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 4);
    ASSERT_EQ(counts[4], 40);
    ASSERT_EQ(counts[5], 20);
    ASSERT_EQ(counts[6], 18);
    ASSERT_EQ(counts[7], 66);
    ASSERT_EQ(counts[8], 52);
    ASSERT_EQ(counts[9], 74);
    ASSERT_EQ(counts[10], 106);
    ASSERT_EQ(counts[11], 98);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 311);
    ASSERT_EQ(counts[1], 41);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 14);
    ASSERT_EQ(counts[4], 23);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 23);
    ASSERT_EQ(counts[7], 18);
    ASSERT_EQ(counts[8], 40);
    ASSERT_EQ(counts[9], 25);
    ASSERT_EQ(counts[10], 56);
    ASSERT_EQ(counts[11], 33);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 154);
    ASSERT_EQ(counts[1], 22);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 8);
    ASSERT_EQ(counts[4], 6);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 6);
    ASSERT_EQ(counts[7], 12);
    ASSERT_EQ(counts[8], 14);
    ASSERT_EQ(counts[9], 10);
    ASSERT_EQ(counts[10], 12);
    ASSERT_EQ(counts[11], 6);
    ASSERT_EQ(counts[12], 4);
}

TEST_F(ArcLengthHistogramTest, testWidthFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 1.0);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 13);
    ASSERT_EQ(counts.size(), 13);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 1.0, 1e-6);
    ASSERT_NEAR(borders[2], 2.0, 1e-6);
    ASSERT_NEAR(borders[3], 3.0, 1e-6);
    ASSERT_NEAR(borders[4], 4.0, 1e-6);
    ASSERT_NEAR(borders[5], 5.0, 1e-6);
    ASSERT_NEAR(borders[6], 6.0, 1e-6);
    ASSERT_NEAR(borders[7], 7.0, 1e-6);
    ASSERT_NEAR(borders[8], 8.0, 1e-6);
    ASSERT_NEAR(borders[9], 9.0, 1e-6);
    ASSERT_NEAR(borders[10], 10.0, 1e-6);
    ASSERT_NEAR(borders[11], 11.0, 1e-6);
    ASSERT_NEAR(borders[12], 12.0, 1e-6);

    ASSERT_EQ(counts[0], 650);
    ASSERT_EQ(counts[1], 110);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 4);
    ASSERT_EQ(counts[4], 40);
    ASSERT_EQ(counts[5], 20);
    ASSERT_EQ(counts[6], 18);
    ASSERT_EQ(counts[7], 66);
    ASSERT_EQ(counts[8], 52);
    ASSERT_EQ(counts[9], 74);
    ASSERT_EQ(counts[10], 106);
    ASSERT_EQ(counts[11], 98);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testCountStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 11.0, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 11.0 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 11.0 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 20);
    ASSERT_EQ(counts[1], 5);
    ASSERT_EQ(counts[2], 20);
}

TEST_F(ArcLengthHistogramTest, testCountStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 11.0, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 11.0 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 11.0 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 12);
    ASSERT_EQ(counts[1], 8);
    ASSERT_EQ(counts[2], 24);
}

TEST_F(ArcLengthHistogramTest, testCountFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 14.0, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 14.0 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 14.0 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 28);
    ASSERT_EQ(counts[1], 54);
    ASSERT_EQ(counts[2], 18);
}

TEST_F(ArcLengthHistogramTest, testCountStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 15.6, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 15.6 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 15.6 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 389);
    ASSERT_EQ(counts[1], 152);
    ASSERT_EQ(counts[2], 67);
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 13.8, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 13.8 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 13.8 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 190);
    ASSERT_EQ(counts[1], 36);
    ASSERT_EQ(counts[2], 30);
}

TEST_F(ArcLengthHistogramTest, testCountFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 17.5, 3);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 17.5 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 17.5 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 824);
    ASSERT_EQ(counts[1], 368);
    ASSERT_EQ(counts[2], 68);
}

TEST_F(ArcLengthHistogramTest, testCountStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 15.6, 3);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 15.6 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 15.6 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 389);
    ASSERT_EQ(counts[1], 152);
    ASSERT_EQ(counts[2], 67);
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 13.8, 3);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 13.8 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 13.8 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 190);
    ASSERT_EQ(counts[1], 36);
    ASSERT_EQ(counts[2], 30);
}

TEST_F(ArcLengthHistogramTest, testCountFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 17.5, 3);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_NEAR(borders[0], 0.0, 1e-6);
    ASSERT_NEAR(borders[1], 17.5 / 3.0, 1e-6);
    ASSERT_NEAR(borders[2], 17.5 / 3.0 * 2.0, 1e-6);

    ASSERT_EQ(counts[0], 824);
    ASSERT_EQ(counts[1], 368);
    ASSERT_EQ(counts[2], 68);
}
