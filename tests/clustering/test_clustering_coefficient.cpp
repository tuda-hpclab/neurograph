/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_clustering_coefficient.h"

#include "metrics/local_structure/Clustering.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(ClusteringCoefficientTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_clustering_coefficient, 0.4119047619, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.20833333333, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_clustering_coefficient, 0.46333333333, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.46333333333, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values come from manually counting
    ASSERT_NEAR(average_clustering_coefficient, 1.0, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 1.0, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_clustering_coefficient, 0.17539682539682540, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.079999999999999988, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_clustering_coefficient, 0.18952380952380951, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.18952380952380951, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_clustering_coefficient, 1.0, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 1.0, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values serve as standard for the four-rank test
        ASSERT_NEAR(average_clustering_coefficient, 0.17539682539682540, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 0.079999999999999988, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values serve as standard for the four-rank test
        ASSERT_NEAR(average_clustering_coefficient, 0.18952380952380951, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 0.18952380952380951, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values serve as standard for the four-rank test
        ASSERT_NEAR(average_clustering_coefficient, 1.0, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 1.0, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // Averaging over 70 nodes of which the ten of the source/sink block close no triangle at all
    // pulls both values below their four-rank counterparts.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_clustering_coefficient, 0.15624986124986123, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.068571428571428561, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_clustering_coefficient, 0.18829931972789113, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.18829931972789113, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // In the complete graph every neighborhood is complete as well, so every node reaches one.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(average_clustering_coefficient, 1.0, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 1.0, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_clustering_coefficient, 0.15624986124986123, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 0.068571428571428561, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_clustering_coefficient, 0.18829931972789113, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 0.18829931972789113, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_clustering_coefficient, 1.0, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 1.0, 1e-6);
    }
}
