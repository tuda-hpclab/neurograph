/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_length_histogram.h"

#include "metrics/geometry/ArcLengthHistogram.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <iostream>
#include <vector>

TEST_F(ArcLengthHistogramTest, testWidthStandard) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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

    ASSERT_EQ(counts[0], 314);
    ASSERT_EQ(counts[1], 47);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 14);
    ASSERT_EQ(counts[4], 23);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 23);
    ASSERT_EQ(counts[7], 18);
    ASSERT_EQ(counts[8], 40);
    ASSERT_EQ(counts[9], 28);
    ASSERT_EQ(counts[10], 56);
    ASSERT_EQ(counts[11], 33);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(4)) {
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

    ASSERT_EQ(counts[0], 314);
    ASSERT_EQ(counts[1], 47);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 14);
    ASSERT_EQ(counts[4], 23);
    ASSERT_EQ(counts[5], 2);
    ASSERT_EQ(counts[6], 23);
    ASSERT_EQ(counts[7], 18);
    ASSERT_EQ(counts[8], 40);
    ASSERT_EQ(counts[9], 28);
    ASSERT_EQ(counts[10], 56);
    ASSERT_EQ(counts[11], 33);
    ASSERT_EQ(counts[12], 22);
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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

    ASSERT_EQ(counts[0], 398);
    ASSERT_EQ(counts[1], 155);
    ASSERT_EQ(counts[2], 67);
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(4)) {
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

    ASSERT_EQ(counts[0], 398);
    ASSERT_EQ(counts[1], 155);
    ASSERT_EQ(counts[2], 67);
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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

TEST_F(ArcLengthHistogramTest, testCountThrowsOnOutOfRangeLength) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    // The longest arc is about 10.86, so a maximum of 5.0 must make the histogram throw.
    ASSERT_ANY_THROW((void) ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 5.0, 5));

    // The shortest arc has length 0.0, so a minimum of 1.0 must make the histogram throw.
    ASSERT_ANY_THROW((void) ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 1.0, 12.0, 11));
}

namespace {
/**
 * @brief Checks the borders and the counts of one arc length histogram against the expected vectors.
 * @param histogram The histogram to check
 * @param expected_borders The expected lower bin borders
 * @param expected_counts The expected number of arcs per bin
 */
template <typename HistogramType>
void expect_length_histogram(const HistogramType& histogram, const std::vector<double>& expected_borders,
                             const std::vector<std::size_t>& expected_counts) {
    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), expected_borders.size());
    ASSERT_EQ(counts.size(), expected_counts.size());

    for (auto bin = std::size_t{ 0 }; bin < expected_counts.size(); ++bin) {
        ASSERT_NEAR(borders[bin], expected_borders[bin], 1e-6) << "at the bin " << bin;
        ASSERT_EQ(counts[bin], expected_counts[bin]) << "at the bin " << bin;
    }
}

// The blocks of the ranks 2, 5, and 6 sit twelve units away from the origin, so the arcs into and
// out of them reach up to 24.9 units and all three seven-rank graphs fill thirteen two-unit bins
const auto wide_borders = std::vector<double>{ 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0 };
const auto count_borders = std::vector<double>{ 0.0, 8.5, 17.0 };
} // namespace

TEST_F(ArcLengthHistogramTest, testWidthStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The 940 data points are the arcs counted by their weight, so the ring arcs of weight 11
    // dominate the picture.
    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, wide_borders, { 312, 28, 39, 27, 92, 101, 132, 33, 66, 0, 44, 44, 22 });
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, wide_borders, { 212, 16, 14, 26, 40, 22, 22, 4, 6, 0, 4, 4, 2 });
}

TEST_F(ArcLengthHistogramTest, testWidthFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, wide_borders, { 926, 8, 118, 152, 214, 758, 790, 352, 252, 62, 174, 72, 28 });
}

TEST_F(ArcLengthHistogramTest, testCountStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, count_borders, { 460, 370, 110 });
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, count_borders, { 284, 78, 10 });
}

TEST_F(ArcLengthHistogramTest, testCountFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_length_histogram(histogram, count_borders, { 1272, 2184, 450 });
}

TEST_F(ArcLengthHistogramTest, testWidthStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, wide_borders, { 312, 28, 39, 27, 92, 101, 132, 33, 66, 0, 44, 44, 22 });
}

TEST_F(ArcLengthHistogramTest, testWidthStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, wide_borders, { 212, 16, 14, 26, 40, 22, 22, 4, 6, 0, 4, 4, 2 });
}

TEST_F(ArcLengthHistogramTest, testWidthFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_width(graph, 2.0);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, wide_borders, { 926, 8, 118, 152, 214, 758, 790, 352, 252, 62, 174, 72, 28 });
}

TEST_F(ArcLengthHistogramTest, testCountStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, count_borders, { 460, 370, 110 });
}

TEST_F(ArcLengthHistogramTest, testCountStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, count_borders, { 284, 78, 10 });
}

TEST_F(ArcLengthHistogramTest, testCountFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto histogram = ArcLengthHistogram::compute_histogram_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_length_histogram(histogram, count_borders, { 1272, 2184, 450 });
}
