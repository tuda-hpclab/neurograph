/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_delta_stepping.h"

#include "Types.h"

#include "metrics/DeltaStepping.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <array>
#include <iostream>

INSTANTIATE_TEST_SUITE_P(DeltaTests, DeltaSteppingTest,
                         testing::Combine(testing::Values(1, 4, 10, 10000),
                                          testing::Values(DeltaSteppingEpochType::PushBruteForce, DeltaSteppingEpochType::PushCheck,
                                                          DeltaSteppingEpochType::PushShortLong, DeltaSteppingEpochType::PushInnerShort,
                                                          DeltaSteppingEpochType::PullModel)));

TEST_P(DeltaSteppingTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    constexpr auto array_size = 10U;
    // These values come from the local results of the APSP implementation
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 25, 31, 24, 22, 20, 45, 32, 31, 23, 26 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 5, 6, 4, 4, 4, 7, 6, 5, 4, 5 };
    constexpr auto efficiencies = std::array<double, array_size>{ 4.3666666, 3.4833333, 4.0, 4.6666666, 4.9166666, 2.1023809, 3.2833333, 3.3166666, 4.0833333, 3.7833333 };

    const auto graph = get_standard_one_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    constexpr auto array_size = 10U;
    // These values come from the local results of the APSP implementation
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 16, 13, 16, 13, 13, 15, 13, 17, 14, 14 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 3, 2, 3, 2, 2, 2, 2, 3, 2, 3 };
    constexpr auto efficiencies = std::array<double, array_size>{ 6.1666666, 7.0, 5.8333333, 7.0, 7.0, 6.0, 7.0, 5.6666666, 6.5, 6.8333333 };

    const auto graph = get_standard_uu_one_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 10 }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        // The values are clear
        ASSERT_EQ(sum_of_distances, 9);
        ASSERT_EQ(maximum_distance, 1);
        ASSERT_NEAR(efficiency, 9.0, 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    constexpr auto array_size = 40U;
    // The values serve as standard for the four-rank test
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 760, 784, 756, 748, 740, 840, 788, 784, 752, 764, 760, 784, 756, 748, 740, 840, 788, 784, 752, 764,
                                                                              760, 784, 756, 748, 740, 840, 788, 784, 752, 764, 760, 784, 756, 748, 740, 840, 788, 784, 752, 764 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 38, 39, 37, 37, 37, 40, 39, 38, 37, 38, 38, 39, 37, 37, 37, 40, 39, 38, 37, 38,
                                                                              38, 39, 37, 37, 37, 40, 39, 38, 37, 38, 38, 39, 37, 37, 37, 40, 39, 38, 37, 38 };
    constexpr auto efficiencies = std::array<double, array_size>{
        5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134, 3.4097103354784561, 4.6742945932261470,
        4.7133019123031259, 5.5366875043729253, 5.2153791587224756, 5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732,
        6.3949926703635134, 3.4097103354784561, 4.6742945932261470, 4.7133019123031259, 5.5366875043729253, 5.2153791587224756, 5.8086783847071217,
        4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134, 3.4097103354784561, 4.6742945932261470, 4.7133019123031259,
        5.5366875043729253, 5.2153791587224756, 5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134,
        3.4097103354784561, 4.6742945932261470, 4.7133019123031259, 5.5366875043729253, 5.2153791587224756
    };

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    constexpr auto array_size = 40U;
    // The values serve as standard for the four-rank test
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 104, 92, 104, 92, 92, 100, 92, 108, 96, 96, 104, 92, 104, 92, 92, 100, 92, 108, 96, 96,
                                                                              104, 92, 104, 92, 92, 100, 92, 108, 96, 96, 104, 92, 104, 92, 92, 100, 92, 108, 96, 96 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 5, 4, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 5, 4, 5,
                                                                              5, 4, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 5, 4, 5 };
    constexpr auto efficiencies = std::array<double, array_size>{ 18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999 };

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ 36 }; node_id++) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { 0, node_id }, delta, epoch_type);

        // The values are clear
        ASSERT_EQ(sum_of_distances, 35);
        ASSERT_EQ(maximum_distance, 1);
        ASSERT_NEAR(efficiency, 35.0, 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    constexpr auto array_size = 40U;
    // The values are the corresponding ones from the one-rank version
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 760, 784, 756, 748, 740, 840, 788, 784, 752, 764, 760, 784, 756, 748, 740, 840, 788, 784, 752, 764,
                                                                              760, 784, 756, 748, 740, 840, 788, 784, 752, 764, 760, 784, 756, 748, 740, 840, 788, 784, 752, 764 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 38, 39, 37, 37, 37, 40, 39, 38, 37, 38, 38, 39, 37, 37, 37, 40, 39, 38, 37, 38,
                                                                              38, 39, 37, 37, 37, 40, 39, 38, 37, 38, 38, 39, 37, 37, 37, 40, 39, 38, 37, 38 };
    constexpr auto efficiencies = std::array<double, array_size>{
        5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134, 3.4097103354784561, 4.6742945932261470,
        4.7133019123031259, 5.5366875043729253, 5.2153791587224756, 5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732,
        6.3949926703635134, 3.4097103354784561, 4.6742945932261470, 4.7133019123031259, 5.5366875043729253, 5.2153791587224756, 5.8086783847071217,
        4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134, 3.4097103354784561, 4.6742945932261470, 4.7133019123031259,
        5.5366875043729253, 5.2153791587224756, 5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134,
        3.4097103354784561, 4.6742945932261470, 4.7133019123031259, 5.5366875043729253, 5.2153791587224756
    };

    const auto graph = get_standard_four_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto actual_node_id = node_id % 10;
        const auto rank = node_id / 10;

        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { utility::save_cast<mpi_rank_type>(rank), actual_node_id }, delta, epoch_type);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            continue;
        }

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    constexpr auto array_size = 40U;
    // The values are the corresponding ones from the one-rank version
    constexpr auto sums_of_distances = std::array<distance_type, array_size>{ 104, 92, 104, 92, 92, 100, 92, 108, 96, 96, 104, 92, 104, 92, 92, 100, 92, 108, 96, 96,
                                                                              104, 92, 104, 92, 92, 100, 92, 108, 96, 96, 104, 92, 104, 92, 92, 100, 92, 108, 96, 96 };
    constexpr auto maximum_distances = std::array<distance_type, array_size>{ 5, 4, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 5, 4, 5,
                                                                              5, 4, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 5, 4, 5 };
    constexpr auto efficiencies = std::array<double, array_size>{ 18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999,
                                                                  18.150000000000002, 19.833333333333336,
                                                                  17.616666666666667, 19.833333333333332,
                                                                  19.833333333333332, 18.0,
                                                                  19.833333333333332, 17.233333333333334,
                                                                  18.916666666666668, 19.449999999999999 };

    const auto graph = get_standard_uu_four_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto actual_node_id = node_id % 10;
        const auto rank = node_id / 10;

        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { utility::save_cast<mpi_rank_type>(rank), actual_node_id }, delta, epoch_type);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            continue;
        }

        ASSERT_EQ(sum_of_distances, sums_of_distances[node_id]);
        ASSERT_EQ(maximum_distance, maximum_distances[node_id]);
        ASSERT_NEAR(efficiency, efficiencies[node_id], 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}

TEST_P(DeltaSteppingTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto& [delta, epoch_type] = GetParam();

    constexpr auto array_size = 36U;

    const auto node_ids = std::array<node_id_type, array_size>{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const auto ranks = std::array<mpi_rank_type, array_size>{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };

    for (auto node_id = node_id_type{ 0 }; node_id < node_id_type{ array_size }; node_id++) {
        const auto actual_node_id = node_ids[node_id];
        const auto rank = ranks[node_id];

        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, { rank, actual_node_id }, delta, epoch_type);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            continue;
        }

        ASSERT_EQ(sum_of_distances, 35);
        ASSERT_EQ(maximum_distance, 1);
        ASSERT_NEAR(efficiency, 35.0, 1e-6);
        ASSERT_EQ(number_unreachable_nodes, 0);
    }
}
