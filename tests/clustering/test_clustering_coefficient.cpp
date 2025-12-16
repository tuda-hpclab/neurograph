/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_clustering_coefficient.h"

#include "metrics/Clustering.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(ClusteringCoefficientTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_clustering_coefficient, 0.15539682539682539, 1e-6);
    ASSERT_NEAR(average_clustering_coefficient_2, 0.058333333333333327, 1e-6);
}

TEST_F(ClusteringCoefficientTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_clustering_coefficient = Clustering::compute_average_clustering_coefficient(graph);
    const auto average_clustering_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values serve as standard for the four-rank test
        ASSERT_NEAR(average_clustering_coefficient, 0.15539682539682539, 1e-6);
        ASSERT_NEAR(average_clustering_coefficient_2, 0.058333333333333327, 1e-6);
    }
}

TEST_F(ClusteringCoefficientTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
