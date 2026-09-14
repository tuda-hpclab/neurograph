/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_reciprocity.h"

#include "metrics/local_structure/Reciprocity.h"

#include <mpi-wrapper/core/MPIInfo.h>

TEST_F(ReciprocityTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // The graph has 25 arcs, two of which are self arcs (5 -> 5 and 6 -> 6) and therefore ignored.
    // Of the remaining 23 arcs, only 2 <-> 4 and 3 <-> 7 are mutual, i.e., 4 arcs are reciprocated.
    // The density is 23 / (10 * 9), so the coefficient is (4 / 23 - 23 / 90) / (1 - 23 / 90) = -169 / 1541
    ASSERT_NEAR(reciprocity, 0.17391304347826086, 1e-6);
    ASSERT_NEAR(coefficient, -0.10966904607397794, 1e-6);
}

TEST_F(ReciprocityTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // Every arc of a symmetric graph is mutual, so the reciprocity is one and, because the graph
    // is not complete (42 of the 90 possible arcs exist), the coefficient reaches its maximum as well
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 1.0, 1e-6);
}

TEST_F(ReciprocityTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // All 90 arcs between distinct nodes exist, so every arc is mutual. The density is one,
    // which makes the coefficient undefined; it is reported as zero
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 0.0, 1e-6);
}

TEST_F(ReciprocityTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // Without arcs there is nothing that could be reciprocated
    ASSERT_NEAR(reciprocity, 0.0, 1e-6);
    ASSERT_NEAR(coefficient, 0.0, 1e-6);
}

TEST_F(ReciprocityTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // The graph consists of four copies of the standard graph, each with 23 arcs between distinct
    // nodes of which 4 are mutual, plus 10 arcs into the next copy that are never mutual (they always
    // point in the same direction around the four copies). That makes 16 of 132 arcs mutual.
    // The density is 132 / (40 * 39), so the coefficient is (4 / 33 - 132 / 1560) / (1 - 132 / 1560).
    // The values serve as standard for the four-rank test
    ASSERT_NEAR(reciprocity, 0.12121212121212122, 1e-6);
    ASSERT_NEAR(coefficient, 0.03997962821492233, 1e-6);
}

TEST_F(ReciprocityTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 1.0, 1e-6);
}

TEST_F(ReciprocityTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 0.0, 1e-6);
}

TEST_F(ReciprocityTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 0.12121212121212122, 1e-6);
        ASSERT_NEAR(coefficient, 0.03997962821492233, 1e-6);
    }
}

TEST_F(ReciprocityTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 1.0, 1e-6);
        ASSERT_NEAR(coefficient, 1.0, 1e-6);
    }
}

TEST_F(ReciprocityTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 1.0, 1e-6);
        ASSERT_NEAR(coefficient, 0.0, 1e-6);
    }
}

TEST_F(ReciprocityTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // Six of the seven blocks contribute 23 arcs between distinct nodes of which 4 are mutual (138
    // and 24); the ring of the four-block component adds 40 never-mutual arcs, the two-block ring 20
    // arcs that are all mutual (its two blocks are each other's predecessor and successor), and the
    // source/sink block 4 arcs. That makes 44 of 202 arcs mutual at a density of 202 / (70 * 69).
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(reciprocity, 0.21782178217821782, 1e-6);
    ASSERT_NEAR(coefficient, 0.18368176489213314, 1e-6);
}

TEST_F(ReciprocityTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 1.0, 1e-6);
}

TEST_F(ReciprocityTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    // All 3906 arcs between distinct nodes exist, so every arc is mutual. The density is one,
    // which makes the coefficient undefined; it is reported as zero.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(reciprocity, 1.0, 1e-6);
    ASSERT_NEAR(coefficient, 0.0, 1e-6);
}

TEST_F(ReciprocityTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 0.21782178217821782, 1e-6);
        ASSERT_NEAR(coefficient, 0.18368176489213314, 1e-6);
    }
}

TEST_F(ReciprocityTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 1.0, 1e-6);
        ASSERT_NEAR(coefficient, 1.0, 1e-6);
    }
}

TEST_F(ReciprocityTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto& [reciprocity, coefficient] = Reciprocity::compute_reciprocity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(reciprocity, 1.0, 1e-6);
        ASSERT_NEAR(coefficient, 0.0, 1e-6);
    }
}
