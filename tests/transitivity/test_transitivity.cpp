/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_transitivity.h"

#include "metrics/local_structure/Transitivity.h"

#include <mpi-wrapper/core/MPIInfo.h>

TEST_F(TransitivityTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The nodes of the graph close 18 of the 46 triangles i->j->k->i that their neighborhoods allow,
    // and 7 of the 36 triangles i->j->k<-i. Unlike the average clustering coefficient of the same
    // graph (0.41190476 and 0.20833333), the two sums are divided instead of the per-node ratios
    ASSERT_NEAR(transitivity, 0.391304347826087, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.19444444444444445, 1e-6);
}

TEST_F(TransitivityTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // Both variants see the same triangles on a symmetric graph, where they are the classic
    // transitivity 3 * (number of triangles) / (number of triples): the graph has 11 triangles, which
    // each of their three nodes closes in both directions (66), and 142 = sum of d * (d - 1) triples
    ASSERT_NEAR(transitivity, 0.4647887323943662, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.4647887323943662, 1e-6);
}

TEST_F(TransitivityTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // Every triangle between three distinct nodes exists, so both variants reach their maximum
    ASSERT_NEAR(transitivity, 1.0, 1e-6);
    ASSERT_NEAR(transitivity_2, 1.0, 1e-6);
}

TEST_F(TransitivityTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // Without nodes there is no triangle that could be closed
    ASSERT_NEAR(transitivity, 0.0, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.0, 1e-6);
}

TEST_F(TransitivityTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The graph consists of four copies of the standard graph, connected by arcs into the next copy;
    // those arcs add triangles to the denominators (72 of 408 and 28 of 328) but close none, because
    // a triangle would have to run back into the previous copy.
    // The values serve as standard for the four-rank test
    ASSERT_NEAR(transitivity, 0.17647058823529413, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.08536585365853659, 1e-6);
}

TEST_F(TransitivityTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The symmetric graph closes 264 of its 1320 triples, i.e., exactly one fifth.
    // The values serve as standard for the four-rank test
    ASSERT_NEAR(transitivity, 0.2, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.2, 1e-6);
}

TEST_F(TransitivityTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(transitivity, 1.0, 1e-6);
    ASSERT_NEAR(transitivity_2, 1.0, 1e-6);
}

TEST_F(TransitivityTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 0.17647058823529413, 1e-6);
        ASSERT_NEAR(transitivity_2, 0.08536585365853659, 1e-6);
    }
}

TEST_F(TransitivityTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 0.2, 1e-6);
        ASSERT_NEAR(transitivity_2, 0.2, 1e-6);
    }
}

TEST_F(TransitivityTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 1.0, 1e-6);
        ASSERT_NEAR(transitivity_2, 1.0, 1e-6);
    }
}

TEST_F(TransitivityTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The source/sink block closes no triangle and offers none either, so only the six copies of the
    // standard block and their ring arcs count. The mutual ring arcs of the two-block component leave
    // the first denominator, which lifts the first variant slightly above the four-rank value while
    // the second one, which never looks at in arcs, stays exactly where it was.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(transitivity, 0.18243243243243243, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.08536585365853659, 1e-6);
}

TEST_F(TransitivityTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(transitivity, 0.22322435174746336, 1e-6);
    ASSERT_NEAR(transitivity_2, 0.22322435174746336, 1e-6);
}

TEST_F(TransitivityTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    // The graph is complete, so every triangle it allows exists.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(transitivity, 1.0, 1e-6);
    ASSERT_NEAR(transitivity_2, 1.0, 1e-6);
}

TEST_F(TransitivityTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 0.18243243243243243, 1e-6);
        ASSERT_NEAR(transitivity_2, 0.08536585365853659, 1e-6);
    }
}

TEST_F(TransitivityTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 0.22322435174746336, 1e-6);
        ASSERT_NEAR(transitivity_2, 0.22322435174746336, 1e-6);
    }
}

TEST_F(TransitivityTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto& [transitivity, transitivity_2] = Transitivity::compute_transitivity(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(transitivity, 1.0, 1e-6);
        ASSERT_NEAR(transitivity_2, 1.0, 1e-6);
    }
}
