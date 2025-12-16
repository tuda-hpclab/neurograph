/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_betweenness_centrality.h"

#include "metrics/Centrality.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(BetweennessCentralityTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 13.583333333333332, 1e-6);
}

TEST_F(BetweennessCentralityTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 5.4, 1e-6);
}

TEST_F(BetweennessCentralityTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
}

TEST_F(BetweennessCentralityTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 112.12194180319423, 1e-6);
}

TEST_F(BetweennessCentralityTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 58.6, 1e-6);
}

TEST_F(BetweennessCentralityTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
}

TEST_F(BetweennessCentralityTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 112.12194180319423, 1e-6);
    }
}

TEST_F(BetweennessCentralityTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 58.6, 1e-6);
    }
}

TEST_F(BetweennessCentralityTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(average_betweenness_centrality, 0.0, 1e-6);
    }
}
