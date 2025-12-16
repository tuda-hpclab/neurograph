/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree_histogram.h"

#include "metrics/DegreeHistogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(DegreeHistogramTest, testStandardWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 1);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 6);
    ASSERT_EQ(in_borders.size(), 6);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 1);
    ASSERT_EQ(in_counts[2], 5);
    ASSERT_EQ(in_counts[3], 3);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 1);

    ASSERT_EQ(out_counts.size(), 5);
    ASSERT_EQ(out_borders.size(), 5);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 6);
    ASSERT_EQ(out_counts[3], 3);
    ASSERT_EQ(out_counts[4], 1);
}

TEST_F(DegreeHistogramTest, testStandardUUWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 1);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 7);
    ASSERT_EQ(in_borders.size(), 7);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 0);
    ASSERT_EQ(in_counts[3], 2);
    ASSERT_EQ(in_counts[4], 3);
    ASSERT_EQ(in_counts[5], 4);
    ASSERT_EQ(in_counts[6], 1);

    ASSERT_EQ(out_counts.size(), 7);
    ASSERT_EQ(out_borders.size(), 7);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 2);
    ASSERT_EQ(out_counts[4], 3);
    ASSERT_EQ(out_counts[5], 4);
    ASSERT_EQ(out_counts[6], 1);
}

TEST_F(DegreeHistogramTest, testFullWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 1);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 11);
    ASSERT_EQ(in_borders.size(), 11);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);
    ASSERT_EQ(in_borders[7], 7);
    ASSERT_EQ(in_borders[8], 8);
    ASSERT_EQ(in_borders[9], 9);
    ASSERT_EQ(in_borders[10], 10);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 0);
    ASSERT_EQ(in_counts[3], 0);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 0);
    ASSERT_EQ(in_counts[6], 0);
    ASSERT_EQ(in_counts[7], 0);
    ASSERT_EQ(in_counts[8], 0);
    ASSERT_EQ(in_counts[9], 0);
    ASSERT_EQ(in_counts[10], 10);

    ASSERT_EQ(out_counts.size(), 11);
    ASSERT_EQ(out_borders.size(), 11);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);
    ASSERT_EQ(out_borders[7], 7);
    ASSERT_EQ(out_borders[8], 8);
    ASSERT_EQ(out_borders[9], 9);
    ASSERT_EQ(out_borders[10], 10);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 0);
    ASSERT_EQ(out_counts[4], 0);
    ASSERT_EQ(out_counts[5], 0);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
    ASSERT_EQ(out_counts[8], 0);
    ASSERT_EQ(out_counts[9], 0);
    ASSERT_EQ(out_counts[10], 10);
}

TEST_F(DegreeHistogramTest, testStandardFourRanksDummyWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 3);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 3);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 2);
    ASSERT_EQ(in_borders.size(), 2);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);

    ASSERT_EQ(in_counts[0], 4);
    ASSERT_EQ(in_counts[1], 36);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);

    ASSERT_EQ(out_counts[0], 4);
    ASSERT_EQ(out_counts[1], 36);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksDummyWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 2);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 5);
    ASSERT_EQ(in_borders.size(), 5);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 2);
    ASSERT_EQ(in_borders[2], 4);
    ASSERT_EQ(in_borders[3], 6);
    ASSERT_EQ(in_borders[4], 8);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 8);
    ASSERT_EQ(in_counts[3], 28);
    ASSERT_EQ(in_counts[4], 4);

    ASSERT_EQ(out_counts.size(), 5);
    ASSERT_EQ(out_borders.size(), 5);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 2);
    ASSERT_EQ(out_borders[2], 4);
    ASSERT_EQ(out_borders[3], 6);
    ASSERT_EQ(out_borders[4], 8);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 8);
    ASSERT_EQ(out_counts[3], 28);
    ASSERT_EQ(out_counts[4], 4);
}

TEST_F(DegreeHistogramTest, testFullFourRanksDummyWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 6);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 6);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 6);
    ASSERT_EQ(in_borders.size(), 6);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 6);
    ASSERT_EQ(in_borders[2], 12);
    ASSERT_EQ(in_borders[3], 18);
    ASSERT_EQ(in_borders[4], 24);
    ASSERT_EQ(in_borders[5], 30);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 0);
    ASSERT_EQ(in_counts[3], 0);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 36);

    ASSERT_EQ(out_counts.size(), 6);
    ASSERT_EQ(out_borders.size(), 6);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 6);
    ASSERT_EQ(out_borders[2], 12);
    ASSERT_EQ(out_borders[3], 18);
    ASSERT_EQ(out_borders[4], 24);
    ASSERT_EQ(out_borders[5], 30);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 0);
    ASSERT_EQ(out_counts[4], 0);
    ASSERT_EQ(out_counts[5], 36);
}

TEST_F(DegreeHistogramTest, testStandardFourRanksWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 3);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 3);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 2);
    ASSERT_EQ(in_borders.size(), 2);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);

    ASSERT_EQ(in_counts[0], 4);
    ASSERT_EQ(in_counts[1], 36);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);

    ASSERT_EQ(out_counts[0], 4);
    ASSERT_EQ(out_counts[1], 36);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 2);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 5);
    ASSERT_EQ(in_borders.size(), 5);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 2);
    ASSERT_EQ(in_borders[2], 4);
    ASSERT_EQ(in_borders[3], 6);
    ASSERT_EQ(in_borders[4], 8);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 8);
    ASSERT_EQ(in_counts[3], 28);
    ASSERT_EQ(in_counts[4], 4);

    ASSERT_EQ(out_counts.size(), 5);
    ASSERT_EQ(out_borders.size(), 5);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 2);
    ASSERT_EQ(out_borders[2], 4);
    ASSERT_EQ(out_borders[3], 6);
    ASSERT_EQ(out_borders[4], 8);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 8);
    ASSERT_EQ(out_counts[3], 28);
    ASSERT_EQ(out_counts[4], 4);
}

TEST_F(DegreeHistogramTest, testFullFourRanksWidth) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_width(graph, 6);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_width(graph, 6);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 6);
    ASSERT_EQ(in_borders.size(), 6);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 6);
    ASSERT_EQ(in_borders[2], 12);
    ASSERT_EQ(in_borders[3], 18);
    ASSERT_EQ(in_borders[4], 24);
    ASSERT_EQ(in_borders[5], 30);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 0);
    ASSERT_EQ(in_counts[3], 0);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 36);

    ASSERT_EQ(out_counts.size(), 6);
    ASSERT_EQ(out_borders.size(), 6);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 6);
    ASSERT_EQ(out_borders[2], 12);
    ASSERT_EQ(out_borders[3], 18);
    ASSERT_EQ(out_borders[4], 24);
    ASSERT_EQ(out_borders[5], 30);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 0);
    ASSERT_EQ(out_counts[4], 0);
    ASSERT_EQ(out_counts[5], 36);
}

TEST_F(DegreeHistogramTest, testStandardCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 7, 7);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 7, 7);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 7);
    ASSERT_EQ(in_borders.size(), 7);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 1);
    ASSERT_EQ(in_counts[2], 5);
    ASSERT_EQ(in_counts[3], 3);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 1);
    ASSERT_EQ(in_counts[6], 0);

    ASSERT_EQ(out_counts.size(), 7);
    ASSERT_EQ(out_borders.size(), 7);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 6);
    ASSERT_EQ(out_counts[3], 3);
    ASSERT_EQ(out_counts[4], 1);
    ASSERT_EQ(out_counts[5], 0);
    ASSERT_EQ(out_counts[6], 0);
}

TEST_F(DegreeHistogramTest, testStandardUUCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 8, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 8, 4);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 4);
    ASSERT_EQ(in_borders.size(), 4);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 2);
    ASSERT_EQ(in_borders[2], 4);
    ASSERT_EQ(in_borders[3], 6);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 2);
    ASSERT_EQ(in_counts[2], 7);
    ASSERT_EQ(in_counts[3], 1);

    ASSERT_EQ(out_counts.size(), 4);
    ASSERT_EQ(out_borders.size(), 4);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 2);
    ASSERT_EQ(out_borders[2], 4);
    ASSERT_EQ(out_borders[3], 6);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 2);
    ASSERT_EQ(out_counts[2], 7);
    ASSERT_EQ(out_counts[3], 1);
}

TEST_F(DegreeHistogramTest, testFullCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 12, 12);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 12, 12);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values come from manually counting
    ASSERT_EQ(in_counts.size(), 12);
    ASSERT_EQ(in_borders.size(), 12);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);
    ASSERT_EQ(in_borders[7], 7);
    ASSERT_EQ(in_borders[8], 8);
    ASSERT_EQ(in_borders[9], 9);
    ASSERT_EQ(in_borders[10], 10);
    ASSERT_EQ(in_borders[11], 11);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 0);
    ASSERT_EQ(in_counts[3], 0);
    ASSERT_EQ(in_counts[4], 0);
    ASSERT_EQ(in_counts[5], 0);
    ASSERT_EQ(in_counts[6], 0);
    ASSERT_EQ(in_counts[7], 0);
    ASSERT_EQ(in_counts[8], 0);
    ASSERT_EQ(in_counts[9], 0);
    ASSERT_EQ(in_counts[10], 10);
    ASSERT_EQ(in_counts[11], 0);

    ASSERT_EQ(out_counts.size(), 12);
    ASSERT_EQ(out_borders.size(), 12);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);
    ASSERT_EQ(out_borders[7], 7);
    ASSERT_EQ(out_borders[8], 8);
    ASSERT_EQ(out_borders[9], 9);
    ASSERT_EQ(out_borders[10], 10);
    ASSERT_EQ(out_borders[11], 11);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 0);
    ASSERT_EQ(out_counts[4], 0);
    ASSERT_EQ(out_counts[5], 0);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
    ASSERT_EQ(out_counts[8], 0);
    ASSERT_EQ(out_counts[9], 0);
    ASSERT_EQ(out_counts[10], 10);
    ASSERT_EQ(out_counts[11], 0);
}

TEST_F(DegreeHistogramTest, testStandardFourRanksDummyCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 8, 8);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 8, 8);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 8);
    ASSERT_EQ(in_borders.size(), 8);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);
    ASSERT_EQ(in_borders[7], 7);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 4);
    ASSERT_EQ(in_counts[3], 20);
    ASSERT_EQ(in_counts[4], 12);
    ASSERT_EQ(in_counts[5], 4);
    ASSERT_EQ(in_counts[6], 0);
    ASSERT_EQ(in_counts[7], 0);

    ASSERT_EQ(out_counts.size(), 8);
    ASSERT_EQ(out_borders.size(), 8);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);
    ASSERT_EQ(out_borders[7], 7);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 4);
    ASSERT_EQ(out_counts[3], 20);
    ASSERT_EQ(out_counts[4], 12);
    ASSERT_EQ(out_counts[5], 4);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksDummyCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 12, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 12, 4);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 4);
    ASSERT_EQ(in_borders.size(), 4);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);
    ASSERT_EQ(in_borders[2], 6);
    ASSERT_EQ(in_borders[3], 9);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 8);
    ASSERT_EQ(in_counts[2], 32);
    ASSERT_EQ(in_counts[3], 0);

    ASSERT_EQ(out_counts.size(), 4);
    ASSERT_EQ(out_borders.size(), 4);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);
    ASSERT_EQ(out_borders[2], 6);
    ASSERT_EQ(out_borders[3], 9);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 8);
    ASSERT_EQ(out_counts[2], 32);
    ASSERT_EQ(out_counts[3], 0);
}

TEST_F(DegreeHistogramTest, testFullFourRanksDummyCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 60, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 60, 2);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test
    ASSERT_EQ(in_counts.size(), 2);
    ASSERT_EQ(in_borders.size(), 2);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 30);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 36);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 30);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 36);
}

TEST_F(DegreeHistogramTest, testStandardFourRanksCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 8, 8);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 8, 8);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 8);
    ASSERT_EQ(in_borders.size(), 8);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 1);
    ASSERT_EQ(in_borders[2], 2);
    ASSERT_EQ(in_borders[3], 3);
    ASSERT_EQ(in_borders[4], 4);
    ASSERT_EQ(in_borders[5], 5);
    ASSERT_EQ(in_borders[6], 6);
    ASSERT_EQ(in_borders[7], 7);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 0);
    ASSERT_EQ(in_counts[2], 4);
    ASSERT_EQ(in_counts[3], 20);
    ASSERT_EQ(in_counts[4], 12);
    ASSERT_EQ(in_counts[5], 4);
    ASSERT_EQ(in_counts[6], 0);
    ASSERT_EQ(in_counts[7], 0);

    ASSERT_EQ(out_counts.size(), 8);
    ASSERT_EQ(out_borders.size(), 8);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 1);
    ASSERT_EQ(out_borders[2], 2);
    ASSERT_EQ(out_borders[3], 3);
    ASSERT_EQ(out_borders[4], 4);
    ASSERT_EQ(out_borders[5], 5);
    ASSERT_EQ(out_borders[6], 6);
    ASSERT_EQ(out_borders[7], 7);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 0);
    ASSERT_EQ(out_counts[2], 4);
    ASSERT_EQ(out_counts[3], 20);
    ASSERT_EQ(out_counts[4], 12);
    ASSERT_EQ(out_counts[5], 4);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 12, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 12, 4);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 4);
    ASSERT_EQ(in_borders.size(), 4);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);
    ASSERT_EQ(in_borders[2], 6);
    ASSERT_EQ(in_borders[3], 9);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 8);
    ASSERT_EQ(in_counts[2], 32);
    ASSERT_EQ(in_counts[3], 0);

    ASSERT_EQ(out_counts.size(), 4);
    ASSERT_EQ(out_borders.size(), 4);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);
    ASSERT_EQ(out_borders[2], 6);
    ASSERT_EQ(out_borders[3], 9);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 8);
    ASSERT_EQ(out_counts[2], 32);
    ASSERT_EQ(out_counts[3], 0);
}

TEST_F(DegreeHistogramTest, testFullFourRanksCount) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto in_histogram = DegreeHistogram::compute_in_degree_fixed_bin_count(graph, 0, 60, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = DegreeHistogram::compute_out_degree_fixed_bin_count(graph, 0, 60, 2);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 2);
    ASSERT_EQ(in_borders.size(), 2);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 30);

    ASSERT_EQ(in_counts[0], 0);
    ASSERT_EQ(in_counts[1], 36);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 30);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 36);
}
