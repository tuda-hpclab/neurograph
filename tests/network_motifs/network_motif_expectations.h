#pragma once

/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

/**
 * @brief The three-node motif census of the graphs of GraphTest, shared by the test files of all
 *		NetworkMotifAlgorithm variants.
 *
 *		Every variant computes the same fractions, so each of them is checked against the same
 *		expectations. They live here instead of being repeated per variant because they are long
 *		(fourteen values per graph) and because a change to a graph of GraphTest has to be reflected
 *		in one place only.
 *
 *		The expectations are stored as counts rather than as fractions: the counts are exact integers
 *		that can be checked against a graph by hand, while the fractions are periodic decimals that
 *		say nothing on their own. expect_motifs derives the fractions from them.
 */
namespace network_motif_expectations {

/**
 * @brief How many motifs of each type a graph contains, indexed by the motif number, i.e., index 0
 *		is unused and the indices 1 to 13 hold the counts.
 */
using MotifCensus = std::array<std::uint64_t, 14>;

/**
 * @brief Builds a census from the counts of the thirteen motifs.
 * @param counts The counts of the motifs 1 to 13, in that order
 * @return The census
 */
[[nodiscard]] constexpr MotifCensus make_census(const std::array<std::uint64_t, 13>& counts) {
    auto result = MotifCensus{};

    for (auto motif_type = std::size_t{ 1 }; motif_type < result.size(); ++motif_type) {
        result[motif_type] = counts[motif_type - 1];
    }

    return result;
}

/**
 * @brief Builds the census of a graph in which every triple is a fully connected mutual triangle,
 *		i.e., of a complete graph.
 * @param number_triples The number of node triples of the graph, i.e., n over 3
 * @return The census
 */
[[nodiscard]] constexpr MotifCensus make_complete_census(const std::uint64_t number_triples) {
    auto result = MotifCensus{};
    result[13] = number_triples;
    return result;
}

// The counts below were derived by enumerating all node triples of the corresponding graph of
// GraphTest and classifying each of them by its isomorphism class. The one-rank values agree with
// the ones that were already asserted for NetworkMotifs::compute_network_triple_motifs, which is
// what ties the whole table to the motif numbering of that implementation.

/**
 * @brief GraphTest::get_standard_one_rank_graph, ten nodes with 49 connected triples
 */
inline constexpr auto standard_one_rank = make_census({ 9, 15, 6, 3, 5, 5, 4, 0, 0, 2, 0, 0, 0 });

/**
 * @brief GraphTest::get_standard_uu_one_rank_graph. Symmetrizing the standard graph leaves the same
 *		49 triples, but every arc is now mutual, so only the mutual open triad 9 and the mutual
 *		triangle 13 can occur.
 */
inline constexpr auto standard_uu_one_rank = make_census({ 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 11 });

/**
 * @brief GraphTest::get_full_one_rank_graph, i.e., the complete graph on ten nodes: all of its
 *		ten over three triples are mutual triangles
 */
inline constexpr auto full_one_rank = make_complete_census(120);

/**
 * @brief GraphTest::get_standard_four_rank_graph and its flattened counterpart. Four copies of the
 *		standard graph, chained into a ring by one arc per node, which raises the number of triples
 *		from 4 * 49 to 572.
 */
inline constexpr auto standard_four_ranks = make_census({ 112, 252, 100, 28, 20, 36, 16, 0, 0, 8, 0, 0, 0 });

/**
 * @brief GraphTest::get_standard_uu_four_rank_graph and its flattened counterpart
 */
inline constexpr auto standard_uu_four_ranks = make_census({ 0, 0, 0, 0, 0, 0, 0, 0, 528, 0, 0, 0, 44 });

/**
 * @brief GraphTest::get_full_four_rank_graph and its flattened counterpart, i.e., the complete graph
 *		on the 8 + 9 + 9 + 10 = 36 nodes of the four MPI ranks
 */
inline constexpr auto full_four_ranks = make_complete_census(7140);

/**
 * @brief GraphTest::get_standard_seven_rank_graph and its flattened counterpart. The four-rank ring
 *		of the ranks 0, 1, 4, and 6 contributes the 572 triples above, the two-rank ring of the ranks
 *		2 and 5 contributes 182, and the source/sink block on rank 3 contributes one, which is the
 *		triple 6 -> 3 <- 9 of the two sources that point at the same sink.
 */
inline constexpr auto standard_seven_ranks = make_census({ 131, 282, 112, 72, 30, 84, 24, 0, 8, 12, 0, 0, 0 });

/**
 * @brief GraphTest::get_standard_uu_seven_rank_graph and its flattened counterpart
 */
inline constexpr auto standard_uu_seven_ranks = make_census({ 0, 0, 0, 0, 0, 0, 0, 0, 689, 0, 0, 0, 66 });

/**
 * @brief GraphTest::get_full_seven_rank_graph and its flattened counterpart, i.e., the complete graph
 *		on the 8 + 9 + 8 + 10 + 9 + 9 + 10 = 63 nodes of the seven MPI ranks
 */
inline constexpr auto full_seven_ranks = make_complete_census(39711);

/**
 * @brief Checks the result of NetworkMotifs::compute_network_triple_motifs against a census, i.e.,
 *		that it reports the total number of triples and the fraction of every motif.
 *		Reports every motif that deviates instead of stopping at the first one, which is what makes a
 *		wrong distribution readable; use it as the last statement of a test.
 * @param motifs The result of the algorithm, i.e., the total count and the thirteen fractions
 * @param expected The expected census of the graph
 */
inline void expect_motifs(const std::array<long double, 14>& motifs, const MotifCensus& expected) {
    const auto expected_total = std::accumulate(expected.begin(), expected.end(), std::uint64_t{ 0 });

    // compute_network_triple_motifs accumulates in long double for the extra precision, but
    // EXPECT_NEAR narrows its arguments to double, so convert once and explicitly here
    EXPECT_NEAR(static_cast<double>(motifs[0]), static_cast<double>(expected_total), 1e-6) << "the total number of motifs";

    ASSERT_GT(expected_total, std::uint64_t{ 0 }) << "a graph without a single triple has no motif fractions";

    for (auto motif_type = std::size_t{ 1 }; motif_type < motifs.size(); ++motif_type) {
        const auto expected_fraction = static_cast<double>(expected[motif_type]) / static_cast<double>(expected_total);

        EXPECT_NEAR(static_cast<double>(motifs[motif_type]), expected_fraction, 1e-6) << "the fraction of motif " << motif_type;
    }
}

} // namespace network_motif_expectations
