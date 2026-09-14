/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_degree_histogram.h"

#include "metrics/degree/InDegreeHistogram.h"
#include "metrics/degree/OutDegreeHistogram.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <iostream>
#include <vector>

TEST_F(DegreeHistogramTest, testStandardWidth) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 1);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 1);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 1);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 1);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 3);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 3);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    // The values serve as standard for the four-rank test. Node 6 of every block has an in degree
    // of 6, so the in histogram needs a third bin; no node has an out degree below 3 anymore, so
    // the out histogram keeps its first bin (the bins always start at 0) but it stays empty.
    ASSERT_EQ(in_counts.size(), 3);
    ASSERT_EQ(in_borders.size(), 3);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);
    ASSERT_EQ(in_borders[2], 6);

    ASSERT_EQ(in_counts[0], 4);
    ASSERT_EQ(in_counts[1], 32);
    ASSERT_EQ(in_counts[2], 4);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 40);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksDummyWidth) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 2);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 6);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 6);

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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 3);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 3);

    const auto out_counts = out_histogram.get_counts();
    const auto out_borders = out_histogram.get_borders();

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(in_counts.size(), 3);
    ASSERT_EQ(in_borders.size(), 3);

    ASSERT_EQ(in_borders[0], 0);
    ASSERT_EQ(in_borders[1], 3);
    ASSERT_EQ(in_borders[2], 6);

    ASSERT_EQ(in_counts[0], 4);
    ASSERT_EQ(in_counts[1], 32);
    ASSERT_EQ(in_counts[2], 4);

    ASSERT_EQ(out_counts.size(), 2);
    ASSERT_EQ(out_borders.size(), 2);

    ASSERT_EQ(out_borders[0], 0);
    ASSERT_EQ(out_borders[1], 3);

    ASSERT_EQ(out_counts[0], 0);
    ASSERT_EQ(out_counts[1], 40);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksWidth) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 2);

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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_width(graph, 6);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_width(graph, 6);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 7, 7);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 7, 7);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 4);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 12);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 12);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8);

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
    ASSERT_EQ(in_counts[5], 0);
    ASSERT_EQ(in_counts[6], 4);
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
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 24);
    ASSERT_EQ(out_counts[4], 12);
    ASSERT_EQ(out_counts[5], 4);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksDummyCount) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4);

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
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 60, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 60, 2);

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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8);

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
    ASSERT_EQ(in_counts[5], 0);
    ASSERT_EQ(in_counts[6], 4);
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
    ASSERT_EQ(out_counts[2], 0);
    ASSERT_EQ(out_counts[3], 24);
    ASSERT_EQ(out_counts[4], 12);
    ASSERT_EQ(out_counts[5], 4);
    ASSERT_EQ(out_counts[6], 0);
    ASSERT_EQ(out_counts[7], 0);
}

TEST_F(DegreeHistogramTest, testStandardUUFourRanksCount) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4);

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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto in_histogram = InDegreeHistogram::compute_fixed_bin_count(graph, 0, 60, 2);

    const auto in_counts = in_histogram.get_counts();
    const auto in_borders = in_histogram.get_borders();

    const auto out_histogram = OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 60, 2);

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

TEST_F(DegreeHistogramTest, testFixedBinCountThrowsOnOutOfRangeDegree) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    // In the standard graph the in degrees range in [1, 5] and the out degrees in [2, 4].
    // A maximum that is too small must throw, because a degree reaches the exclusive upper bound ...
    ASSERT_ANY_THROW((void) InDegreeHistogram::compute_fixed_bin_count(graph, 0, 3, 3));
    ASSERT_ANY_THROW((void) OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 3, 3));

    // ... and a minimum that is too large must throw as well, because a degree lies below the inclusive lower bound.
    ASSERT_ANY_THROW((void) InDegreeHistogram::compute_fixed_bin_count(graph, 2, 8, 6));
    ASSERT_ANY_THROW((void) OutDegreeHistogram::compute_fixed_bin_count(graph, 3, 9, 6));
}

namespace {
/**
 * @brief Checks the borders and the counts of one degree histogram against the expected vectors.
 * @param histogram The histogram to check
 * @param expected_borders The expected lower bin borders
 * @param expected_counts The expected number of nodes per bin
 */
template <typename HistogramType>
void expect_histogram(const HistogramType& histogram, const std::vector<arc_id_type>& expected_borders,
                      const std::vector<std::size_t>& expected_counts) {
    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), expected_borders.size());
    ASSERT_EQ(counts.size(), expected_counts.size());

    for (auto bin = std::size_t{ 0 }; bin < expected_counts.size(); ++bin) {
        ASSERT_EQ(borders[bin], expected_borders[bin]) << "at the bin " << bin;
        ASSERT_EQ(counts[bin], expected_counts[bin]) << "at the bin " << bin;
    }
}
// The complete full seven-rank graph is 62-regular, so its width-six histogram fills only the bin
// [60, 66) that holds the degree 62
const auto full_width_borders = std::vector<arc_id_type>{ 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60 };
const auto full_width_counts = std::vector<std::size_t>{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63 };
} // namespace

// The degrees of the standard seven-rank graph are the ones of the standard block plus the single
// ring arc (in: 2, 4, 3, 4, 3, 3, 6, 3, 3, 4; out: 4, 3, 3, 4, 5, 4, 3, 3, 3, 3) in each of the six
// standard blocks, plus the source/sink block, which contributes seven in degrees of zero, two of
// one and one of two, and six out degrees of zero and four of one.

TEST_F(DegreeHistogramTest, testStandardSevenRanksDummyWidth) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 3), { 0, 3, 6 }, { 16, 48, 6 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 3), { 0, 3 }, { 10, 60 });
}

TEST_F(DegreeHistogramTest, testStandardUUSevenRanksDummyWidth) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 2), { 0, 2, 4, 6, 8 }, { 9, 1, 18, 38, 4 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 2), { 0, 2, 4, 6, 8 }, { 9, 1, 18, 38, 4 });
}

TEST_F(DegreeHistogramTest, testFullSevenRanksDummyWidth) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // Every node of the complete graph has 62 in and 62 out arcs, which is the eleventh bin.
    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 6), full_width_borders, full_width_counts);
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 6), full_width_borders, full_width_counts);
}

TEST_F(DegreeHistogramTest, testStandardSevenRanksDummyCount) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8), { 0, 1, 2, 3, 4, 5, 6, 7 },
                     { 7, 2, 7, 30, 18, 0, 6, 0 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8), { 0, 1, 2, 3, 4, 5, 6, 7 },
                     { 6, 4, 0, 36, 18, 6, 0, 0 });
}

TEST_F(DegreeHistogramTest, testStandardUUSevenRanksDummyCount) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4), { 0, 3, 6, 9 }, { 10, 18, 42, 0 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4), { 0, 3, 6, 9 }, { 10, 18, 42, 0 });
}

TEST_F(DegreeHistogramTest, testFullSevenRanksDummyCount) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    // The values serve as standard for the seven-rank test
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 70, 2), { 0, 35 }, { 0, 63 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 70, 2), { 0, 35 }, { 0, 63 });
}

TEST_F(DegreeHistogramTest, testStandardSevenRanksWidth) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 3), { 0, 3, 6 }, { 16, 48, 6 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 3), { 0, 3 }, { 10, 60 });
}

TEST_F(DegreeHistogramTest, testStandardUUSevenRanksWidth) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 2), { 0, 2, 4, 6, 8 }, { 9, 1, 18, 38, 4 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 2), { 0, 2, 4, 6, 8 }, { 9, 1, 18, 38, 4 });
}

TEST_F(DegreeHistogramTest, testFullSevenRanksWidth) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_width(graph, 6), full_width_borders, full_width_counts);
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_width(graph, 6), full_width_borders, full_width_counts);
}

TEST_F(DegreeHistogramTest, testStandardSevenRanksCount) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8), { 0, 1, 2, 3, 4, 5, 6, 7 },
                     { 7, 2, 7, 30, 18, 0, 6, 0 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 8, 8), { 0, 1, 2, 3, 4, 5, 6, 7 },
                     { 6, 4, 0, 36, 18, 6, 0, 0 });
}

TEST_F(DegreeHistogramTest, testStandardUUSevenRanksCount) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4), { 0, 3, 6, 9 }, { 10, 18, 42, 0 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 12, 4), { 0, 3, 6, 9 }, { 10, 18, 42, 0 });
}

TEST_F(DegreeHistogramTest, testFullSevenRanksCount) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    // The values are the corresponding ones from the one-rank version
    expect_histogram(InDegreeHistogram::compute_fixed_bin_count(graph, 0, 70, 2), { 0, 35 }, { 0, 63 });
    expect_histogram(OutDegreeHistogram::compute_fixed_bin_count(graph, 0, 70, 2), { 0, 35 }, { 0, 63 });
}
