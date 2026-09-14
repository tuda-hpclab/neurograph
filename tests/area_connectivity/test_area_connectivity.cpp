/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_area_connectivity.h"

#include "metrics/community/AreaConnectivity.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(AreaConnectivityTest, testStandard) {
    if (skip_unless_rank_count(1)) {
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

TEST_F(AreaConnectivityTest, testEmpty) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_empty_one_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    ASSERT_TRUE(area_connectivity.empty());
}

TEST_F(AreaConnectivityTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 92);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 18);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 66);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 45);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 44);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 58);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 53);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 38);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 34);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 13);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 22);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 25);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(1)) {
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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

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

    ASSERT_EQ(area_connectivity.at({ "area_1", "area_1" }), 61);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_2" }), 17);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_3" }), 4);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_4" }), 92);
    ASSERT_EQ(area_connectivity.at({ "area_1", "area_5" }), 18);

    ASSERT_EQ(area_connectivity.at({ "area_2", "area_1" }), 66);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_2" }), 45);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_3" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_2", "area_4" }), 44);

    ASSERT_EQ(area_connectivity.at({ "area_3", "area_1" }), 11);
    ASSERT_EQ(area_connectivity.at({ "area_3", "area_2" }), 3);

    ASSERT_EQ(area_connectivity.at({ "area_4", "area_1" }), 58);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_2" }), 53);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_4" }), 38);
    ASSERT_EQ(area_connectivity.at({ "area_4", "area_5" }), 34);

    ASSERT_EQ(area_connectivity.at({ "area_5", "area_1" }), 13);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_2" }), 22);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_4" }), 25);
    ASSERT_EQ(area_connectivity.at({ "area_5", "area_5" }), 5);
}

TEST_F(AreaConnectivityTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

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
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

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

namespace {
/**
 * @brief The area connectivity of the standard seven-rank graph. Comparing the whole map at once
 *		also pins down which pairs are absent, which is what distinguishes this graph: area_3 only
 *		exists in the two blocks of canonical rank 0 and never reaches area_4 or area_5.
 */
[[nodiscard]] AreaConnectivityMap get_expected_standard_seven_rank_connectivity() {
    return AreaConnectivityMap{
        { { "area_1", "area_1" }, 118 }, { { "area_1", "area_2" }, 45 }, { { "area_1", "area_3" }, 19 },
        { { "area_1", "area_4" }, 126 }, { { "area_1", "area_5" }, 18 }, { { "area_2", "area_1" }, 110 },
        { { "area_2", "area_2" }, 76 },  { { "area_2", "area_3" }, 11 }, { { "area_2", "area_4" }, 70 },
        { { "area_3", "area_1" }, 22 },  { { "area_3", "area_2" }, 6 },  { { "area_4", "area_1" }, 86 },
        { { "area_4", "area_2" }, 81 },  { { "area_4", "area_4" }, 53 }, { { "area_4", "area_5" }, 34 },
        { { "area_5", "area_1" }, 13 },  { { "area_5", "area_2" }, 22 }, { { "area_5", "area_4" }, 25 },
        { { "area_5", "area_5" }, 5 },
    };
}

[[nodiscard]] AreaConnectivityMap get_expected_standard_uu_seven_rank_connectivity() {
    // Symmetrizing the arcs makes the map symmetric as well
    return AreaConnectivityMap{
        { { "area_1", "area_1" }, 55 }, { { "area_1", "area_2" }, 29 }, { { "area_1", "area_3" }, 8 },
        { { "area_1", "area_4" }, 38 }, { { "area_1", "area_5" }, 9 },  { { "area_2", "area_1" }, 29 },
        { { "area_2", "area_2" }, 31 }, { { "area_2", "area_3" }, 5 },  { { "area_2", "area_4" }, 26 },
        { { "area_2", "area_5" }, 2 },  { { "area_3", "area_1" }, 8 },  { { "area_3", "area_2" }, 5 },
        { { "area_4", "area_1" }, 38 }, { { "area_4", "area_2" }, 26 }, { { "area_4", "area_4" }, 31 },
        { { "area_4", "area_5" }, 8 },  { { "area_5", "area_1" }, 9 },  { { "area_5", "area_2" }, 2 },
        { { "area_5", "area_4" }, 8 },  { { "area_5", "area_5" }, 5 },
    };
}

[[nodiscard]] AreaConnectivityMap get_expected_full_seven_rank_connectivity() {
    // The complete graph connects every node to every other one, so the strength of a pair of
    // distinct areas is the product of their node counts and the map is symmetric. The area sizes
    // are 18 for area_1, 18 for area_2, 2 for area_3, 21 for area_4 and 4 for area_5, so a pair of
    // equal areas keeps n * (n - 1) arcs.
    return AreaConnectivityMap{
        { { "area_1", "area_1" }, 306 }, { { "area_1", "area_2" }, 324 }, { { "area_1", "area_3" }, 36 },
        { { "area_1", "area_4" }, 378 }, { { "area_1", "area_5" }, 72 },  { { "area_2", "area_1" }, 324 },
        { { "area_2", "area_2" }, 306 }, { { "area_2", "area_3" }, 36 },  { { "area_2", "area_4" }, 378 },
        { { "area_2", "area_5" }, 72 },  { { "area_3", "area_1" }, 36 },  { { "area_3", "area_2" }, 36 },
        { { "area_3", "area_3" }, 2 },   { { "area_3", "area_4" }, 42 },  { { "area_3", "area_5" }, 8 },
        { { "area_4", "area_1" }, 378 }, { { "area_4", "area_2" }, 378 }, { { "area_4", "area_3" }, 42 },
        { { "area_4", "area_4" }, 420 }, { { "area_4", "area_5" }, 84 },  { { "area_5", "area_1" }, 72 },
        { { "area_5", "area_2" }, 72 },  { { "area_5", "area_3" }, 8 },   { { "area_5", "area_4" }, 84 },
        { { "area_5", "area_5" }, 12 },
    };
}
} // namespace

TEST_F(AreaConnectivityTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_EQ(area_connectivity, get_expected_standard_seven_rank_connectivity());
}

TEST_F(AreaConnectivityTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_EQ(area_connectivity, get_expected_standard_uu_seven_rank_connectivity());
}

TEST_F(AreaConnectivityTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    // The values serve as standard for the seven-rank test
    ASSERT_EQ(area_connectivity, get_expected_full_seven_rank_connectivity());
}

TEST_F(AreaConnectivityTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version; the area ids are local to
        // their rank here and only the gathered name lists turn them back into these global names
        ASSERT_EQ(area_connectivity, get_expected_standard_seven_rank_connectivity());
    }
}

TEST_F(AreaConnectivityTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_EQ(area_connectivity, get_expected_standard_uu_seven_rank_connectivity());
    }
}

TEST_F(AreaConnectivityTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(graph);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_EQ(area_connectivity, get_expected_full_seven_rank_connectivity());
    }
}
