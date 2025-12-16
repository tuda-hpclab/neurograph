/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_apsp.h"

#include "metrics/AllPairsShortestPath.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(APSPTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 3.1);
    ASSERT_NEAR(average_efficiency, 0.4222486772486772, 1e-6);
    ASSERT_EQ(diameter, 7);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.6, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.7222222222222222, 1e-6);
    ASSERT_EQ(diameter, 3);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_EQ(average_efficiency, 1.0);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 19.784615384615385);
    ASSERT_NEAR(average_efficiency, 0.13387777556694200, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 19.784615384615385);
    ASSERT_NEAR(average_efficiency, 0.13387777556694200, 1e-6);
    ASSERT_EQ(diameter, 40);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPInverseTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.7043209876543211, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.89050315072273489, 1e-6);
    ASSERT_EQ(diameter, 4.5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_NEAR(average_shortest_path_len, 1.6, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.7222222222222222, 1e-6);
    ASSERT_EQ(diameter, 3);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values comes from bctpy
    ASSERT_EQ(average_shortest_path_len, 1.0);
    ASSERT_EQ(average_efficiency, 1.0);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(10));
    ASSERT_EQ(cluster_sizes.at(10), 10);
}

TEST_F(APSPInverseTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 1.7603729603729605, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.1823728020058155, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}

TEST_F(APSPInverseTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 1.7603729603729605, 1e-6);
    ASSERT_NEAR(average_efficiency, 1.1823728020058155, 1e-6);
    ASSERT_NEAR(diameter, 4.7727272727272725, 1e-6);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_NEAR(average_shortest_path_len, 2.5025641025641026, 1e-6);
    ASSERT_NEAR(average_efficiency, 0.48384615384615381, 1e-6);
    ASSERT_EQ(diameter, 5);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(40));
    ASSERT_EQ(cluster_sizes.at(40), 40);
}

TEST_F(APSPInverseTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [average_shortest_path_len, average_efficiency, diameter, number_disconnected_pairs, cluster_sizes] = AllPairsShortestPath::compute_apsp_inverse(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(average_shortest_path_len, 1.0000000000000000);
    ASSERT_NEAR(average_efficiency, 1.0000000000000000, 1e-6);
    ASSERT_EQ(diameter, 1);
    ASSERT_EQ(number_disconnected_pairs, 0);
    ASSERT_EQ(cluster_sizes.size(), 1);
    ASSERT_TRUE(cluster_sizes.contains(36));
    ASSERT_EQ(cluster_sizes.at(36), 36);
}
