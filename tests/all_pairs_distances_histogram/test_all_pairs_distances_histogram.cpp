/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_all_pairs_distances_histogram.h"

#include "metrics/geometry/AllPairsDistancesHistogram.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <iostream>
#include <vector>

TEST_F(AllPairsDistancesHistogramTest, testWidthStandard) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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
    if (skip_unless_rank_count(4)) {
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

TEST_F(AllPairsDistancesHistogramTest, testCountThrowsOnOutOfRangeDistance) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    // The largest pairwise distance is about 11.97, so a maximum of 6.0 must make the histogram throw.
    ASSERT_ANY_THROW((void) AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 6.0, 3));

    // The smallest pairwise distance is about 1.17, so a minimum of 2.0 must make the histogram throw.
    ASSERT_ANY_THROW((void) AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 2.0, 14.0, 6));
}

namespace {
/**
 * @brief Checks the borders and the counts of one pairwise-distance histogram against the expected vectors.
 * @param histogram The histogram to check
 * @param expected_borders The expected lower bin borders
 * @param expected_counts The expected number of node pairs per bin
 */
template <typename HistogramType>
void expect_distance_histogram(const HistogramType& histogram, const std::vector<double>& expected_borders,
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
} // namespace

// The standard and the undirected, unit-weight seven-rank graph hold the same nodes at the same
// positions, so their 2415 pairwise distances agree; the full graph has 63 nodes and 1953 pairs.

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 573, 288, 1033, 303, 218 });
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 573, 288, 1033, 303, 218 });
}

TEST_F(AllPairsDistancesHistogramTest, testWidthFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 495, 214, 860, 247, 137 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 763, 1299, 353 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 763, 1299, 353 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values serve as standard for the seven-rank test
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 636, 1092, 225 });
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 573, 288, 1033, 303, 218 });
}

TEST_F(AllPairsDistancesHistogramTest, testWidthStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 573, 288, 1033, 303, 218 });
}

TEST_F(AllPairsDistancesHistogramTest, testWidthFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(graph, 5.0);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 5.0, 10.0, 15.0, 20.0 }, { 495, 214, 860, 247, 137 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 763, 1299, 353 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 763, 1299, 353 });
}

TEST_F(AllPairsDistancesHistogramTest, testCountFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto histogram = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(graph, 0.0, 25.5, 3);

    // The values are the corresponding ones from the one-rank version
    expect_distance_histogram(histogram, { 0.0, 8.5, 17.0 }, { 636, 1092, 225 });
}
