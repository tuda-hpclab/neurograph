/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_area_connectivity.h"

#include "metrics/AreaConnectivity.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(AreaConnectivityTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values come from manually counting
    ASSERT_EQ(area_connectivity.size(), 6);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 6);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 19);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);
}

TEST_F(AreaConnectivityTest, testStandardUU) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values come from manually counting
    ASSERT_EQ(area_connectivity.size(), 8);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 7);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 7 + 2);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 0 + 3);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 2 + 7);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 2 + 0);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 3 + 0);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 0 + 2);
}

TEST_F(AreaConnectivityTest, testFull) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values come from manually counting
    ASSERT_EQ(area_connectivity.size(), 9);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_3" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 16);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 25);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 5);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 5);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_3" }), 1);
}

TEST_F(AreaConnectivityTest, testStandardFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(area_connectivity.size(), 19);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 61);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 17);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 89);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 18);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 63);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 45);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 44);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 55);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 53);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 35);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 34);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 13);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 22);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 25);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(area_connectivity.size(), 20);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 32);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 16);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 26);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 9);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 16);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 3);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_5" }), 2);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 26);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 23);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 8);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 2);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 8);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testFullFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the four-rank test
    ASSERT_EQ(area_connectivity.size(), 24);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 132);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 108);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 12);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 120);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 48);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 108);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 72);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_5" }), 36);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 12);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_4" }), 10);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_5" }), 4);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 120);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_3" }), 10);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 40);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 48);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 36);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 40);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 12);
}

TEST_F(AreaConnectivityTest, testStandardFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    return;

    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(area_connectivity.size(), 19);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    /*

    area_1 --> area_1: 26
    area_1 --> area_2: 10
    area_1 --> area_3: 5
    area_1 --> area_4: 21
    area_1 --> area_5: 13

    area_2 --> area_1: 24
    area_2 --> area_2: 16
    area_2 --> area_4: 26

    area_3 --> area_2: 3

    area_4 --> area_1: 12
    area_4 --> area_2: 14
    area_4 --> area_4: 28
    area_4 --> area_5: 1

    area_5 --> area_1: 13
    area_5 --> area_4: 3
    area_5 --> area_5: 5

*/

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 61);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 17);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 89);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 18);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 63);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 45);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 44);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 55);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 53);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 35);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 34);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 13);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 22);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 25);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }
    return;
    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(area_connectivity.size(), 20);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 32);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 16);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 26);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 9);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 16);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 3);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_5" }), 2);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 26);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 20);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 23);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 8);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 2);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 8);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testFullFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }
    return;
    // The values are the corresponding ones from the one-rank version
    ASSERT_EQ(area_connectivity.size(), 24);

    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_1", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_2", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_3", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_4", "area_5" }));

    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_1" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_2" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_3" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_4" }));
    ASSERT_TRUE(area_connectivity.contains({ "area_5", "area_5" }));

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 132);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 108);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 12);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 120);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 48);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 108);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 72);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_5" }), 36);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 12);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 9);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_4" }), 10);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_5" }), 4);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 120);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_3" }), 10);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 90);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 40);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 48);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 36);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 40);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 12);
}
