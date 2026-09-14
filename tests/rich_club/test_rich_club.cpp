/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_rich_club.h"

#include "Types.h"

#include "metrics/local_structure/RichClub.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <vector>

namespace {
/**
 * @brief Checks that the result reports exactly the expected degree thresholds, i.e., that all three
 *		of its vectors have the expected length and content.
 * @param result The result to check
 * @param expected_club_sizes The expected club size per degree threshold
 * @param expected_arc_counts The expected number of club arcs per degree threshold
 * @param expected_coefficients The expected coefficient per degree threshold
 */
void expect_rich_club(const RichClubResult& result, const std::vector<global_node_id_type>& expected_club_sizes,
                      const std::vector<distance_type>& expected_arc_counts, const std::vector<double>& expected_coefficients) {
    ASSERT_EQ(result.club_sizes, expected_club_sizes);
    ASSERT_EQ(result.club_arc_counts, expected_arc_counts);
    ASSERT_EQ(result.coefficients.size(), expected_coefficients.size());

    for (auto threshold = std::size_t{ 0 }; threshold < expected_coefficients.size(); ++threshold) {
        ASSERT_NEAR(result.coefficients[threshold], expected_coefficients[threshold], 1e-6) << "at the degree threshold " << threshold;
    }
}
} // namespace

// The degrees are the number of in arcs plus the number of out arcs, so a self arc counts twice. All
// values come from a Python reference that builds the club of every threshold explicitly and counts
// the arcs inside it, i.e., without the histograms and suffix sums of the implementation.

TEST_F(RichClubTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The degrees are 4, 5, 4, 6, 6, 5, 7, 4, 4, 5, so all ten nodes survive the thresholds 0 to 3,
    // the six nodes of a degree above four the threshold 4, and only 3, 4, and 6 the threshold 5.
    // Of the 25 arcs, the two self arcs (5 -> 5 and 6 -> 6) never count, which leaves 23 for the
    // whole graph. The club of the threshold 4 is denser than that (0.3 against 0.256), but the three
    // richest nodes are connected by the single arc 4 -> 6, so the curve falls again at the very top
    expect_rich_club(result, { 10, 10, 10, 10, 6, 3 }, { 23, 23, 23, 23, 9, 1 },
                     { 0.25555555555555554, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554, 0.3, 0.16666666666666666 });
}

TEST_F(RichClubTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // Symmetrizing doubles every degree (8, 10, 6, 10, 10, 8, 12, 6, 8, 10) and every arc count, so
    // the coefficients are the classic undirected ones at half the threshold. Here the club really is
    // a rich one: its density grows from 42 / 90 over the whole graph to 14 / 20 among the five nodes
    // of a degree above eight
    expect_rich_club(result, { 10, 10, 10, 10, 10, 10, 8, 8, 5, 5 }, { 42, 42, 42, 42, 42, 42, 32, 32, 14, 14 },
                     { 0.46666666666666667, 0.46666666666666667, 0.46666666666666667, 0.46666666666666667, 0.46666666666666667, 0.46666666666666667,
                       0.5714285714285714, 0.5714285714285714, 0.7, 0.7 });
}

TEST_F(RichClubTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // Every node has all ten in and all ten out arcs, so all of them belong to every club and all 90
    // arcs between distinct nodes run inside it. The club is complete, which is the maximum density
    expect_rich_club(result, std::vector<global_node_id_type>(20, 10), std::vector<distance_type>(20, 90), std::vector<double>(20, 1.0));
}

TEST_F(RichClubTest, testAcyclic) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_negative_weight_one_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The arcs are 0 -> 1, 0 -> 2, 1 -> 2, 1 -> 3, and 2 -> 3, so the source and the sink have a
    // degree of two while the two nodes in between have one of three. Only 1 -> 2 connects those two,
    // which still is a higher density than the 5 / 12 of the whole graph. The negative weights play
    // no role
    expect_rich_club(result, { 4, 4, 2 }, { 5, 5, 1 }, { 0.41666666666666669, 0.41666666666666669, 0.5 });
}

TEST_F(RichClubTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // Without nodes no threshold has a club of two nodes, so no coefficient is defined
    expect_rich_club(result, {}, {}, {});
}

TEST_F(RichClubTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The graph consists of four copies of the standard graph, and the ring arcs into the next copy
    // raise every degree by two. The 132 arcs between distinct nodes are the 4 * 23 of the copies
    // plus the 40 ring arcs. The club of the threshold 8 holds the four copies of node 6, which the
    // ring connects into a cycle: 4 arcs among 4 nodes.
    // The values serve as standard for the four-rank test
    expect_rich_club(result, { 40, 40, 40, 40, 40, 40, 24, 12, 4 }, { 132, 132, 132, 132, 132, 132, 60, 16, 4 },
                     { 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462,
                       0.10869565217391304, 0.12121212121212122, 0.33333333333333331 });
}

TEST_F(RichClubTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The values serve as standard for the four-rank test
    expect_rich_club(result, { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 32, 32, 20, 20, 4, 4 },
                     { 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 192, 192, 96, 96, 8, 8 },
                     { 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896,
                       0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.19354838709677419, 0.19354838709677419,
                       0.25263157894736843, 0.25263157894736843, 0.66666666666666663, 0.66666666666666663 });
}

TEST_F(RichClubTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // All 36 nodes are connected to all others but not to themselves, so every degree is 70 and all
    // 36 * 35 arcs run inside the only club.
    // The values serve as standard for the four-rank test
    expect_rich_club(result, std::vector<global_node_id_type>(70, 36), std::vector<distance_type>(70, 1260), std::vector<double>(70, 1.0));
}

TEST_F(RichClubTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        expect_rich_club(result, { 40, 40, 40, 40, 40, 40, 24, 12, 4 }, { 132, 132, 132, 132, 132, 132, 60, 16, 4 },
                         { 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462, 0.08461538461538462,
                           0.10869565217391304, 0.12121212121212122, 0.33333333333333331 });
    }
}

TEST_F(RichClubTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        expect_rich_club(result, { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 32, 32, 20, 20, 4, 4 },
                         { 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 192, 192, 96, 96, 8, 8 },
                         { 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896,
                           0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.15897435897435896, 0.19354838709677419, 0.19354838709677419,
                           0.25263157894736843, 0.25263157894736843, 0.66666666666666663, 0.66666666666666663 });
    }
}

TEST_F(RichClubTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        expect_rich_club(result, std::vector<global_node_id_type>(70, 36), std::vector<distance_type>(70, 1260), std::vector<double>(70, 1.0));
    }
}

TEST_F(RichClubTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The three isolated nodes of the source/sink block have a degree of zero and drop out of the
    // club at the very first threshold, which is why it starts at 67 instead of 70. Its sinks and
    // sources have a degree of one or two and follow immediately; from the threshold 2 on the club
    // is exactly the 60 nodes of the six standard blocks.
    // The values serve as standard for the seven-rank test
    expect_rich_club(result, { 67, 61, 60, 60, 60, 60, 36, 18, 6 }, { 202, 198, 198, 198, 198, 198, 90, 24, 6 },
                     { 0.045680687471732250, 0.054098360655737705, 0.055932203389830504, 0.055932203389830504, 0.055932203389830504,
                       0.055932203389830504, 0.071428571428571425, 0.078431372549019607, 0.20 });
}

TEST_F(RichClubTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // The values serve as standard for the seven-rank test
    expect_rich_club(result, { 67, 67, 61, 61, 60, 60, 60, 60, 56, 56, 42, 42, 22, 22, 4, 4 },
                     { 360, 360, 352, 352, 352, 352, 352, 352, 328, 328, 230, 230, 98, 98, 8, 8 },
                     { 0.081411126187245599, 0.081411126187245599, 0.096174863387978142, 0.096174863387978142, 0.099435028248587576,
                       0.099435028248587576, 0.099435028248587576, 0.099435028248587576, 0.106493506493506490, 0.106493506493506490,
                       0.133565621370499430, 0.133565621370499430, 0.212121212121212130, 0.212121212121212130, 0.66666666666666663,
                       0.66666666666666663 });
}

TEST_F(RichClubTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    // Every node has 62 in and 62 out arcs, so no threshold up to the last one of 123 excludes a
    // node: the club always holds all 63 nodes and all 3906 arcs, which is a coefficient of one.
    const auto expected_club_sizes = std::vector<global_node_id_type>(124, global_node_id_type{ 63 });
    const auto expected_arc_counts = std::vector<distance_type>(124, distance_type{ 3906 });
    const auto expected_coefficients = std::vector<double>(124, 1.0);

    // The values serve as standard for the seven-rank test
    expect_rich_club(result, expected_club_sizes, expected_arc_counts, expected_coefficients);
}

TEST_F(RichClubTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        expect_rich_club(result, { 67, 61, 60, 60, 60, 60, 36, 18, 6 }, { 202, 198, 198, 198, 198, 198, 90, 24, 6 },
                         { 0.045680687471732250, 0.054098360655737705, 0.055932203389830504, 0.055932203389830504, 0.055932203389830504,
                           0.055932203389830504, 0.071428571428571425, 0.078431372549019607, 0.20 });
    }
}

TEST_F(RichClubTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        expect_rich_club(result, { 67, 67, 61, 61, 60, 60, 60, 60, 56, 56, 42, 42, 22, 22, 4, 4 },
                         { 360, 360, 352, 352, 352, 352, 352, 352, 328, 328, 230, 230, 98, 98, 8, 8 },
                         { 0.081411126187245599, 0.081411126187245599, 0.096174863387978142, 0.096174863387978142, 0.099435028248587576,
                           0.099435028248587576, 0.099435028248587576, 0.099435028248587576, 0.106493506493506490, 0.106493506493506490,
                           0.133565621370499430, 0.133565621370499430, 0.212121212121212130, 0.212121212121212130, 0.66666666666666663,
                           0.66666666666666663 });
    }
}

TEST_F(RichClubTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto result = RichClub::compute_rich_club_coefficients(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    const auto expected_club_sizes = std::vector<global_node_id_type>(124, global_node_id_type{ 63 });
    const auto expected_arc_counts = std::vector<distance_type>(124, distance_type{ 3906 });
    const auto expected_coefficients = std::vector<double>(124, 1.0);

    // The values are the corresponding ones from the one-rank version
    expect_rich_club(result, expected_club_sizes, expected_arc_counts, expected_coefficients);
}
