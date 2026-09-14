/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_assortativity.h"

#include "metrics/local_structure/Assortativity.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <iostream>

TEST_F(AssortativityTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The value come from computing the Pearson correlation coefficient by hand
    // https://wikimedia.org/api/rest_v1/media/math/render/svg/5984dfb290912b0e0b92a984bf49cdd628c38b2c
    // because bctpy computes bogus
    ASSERT_NEAR(r_in_in, 0.19335037815687611, 1e-6);
    ASSERT_NEAR(r_in_out, -0.11870884189326848, 1e-6);
    ASSERT_NEAR(r_out_in, -0.016243354311358746, 1e-6);
    ASSERT_NEAR(r_out_out, 0.48540898159236057, 1e-6);
}

TEST_F(AssortativityTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The value come from computing the Pearson correlation coefficient by hand
    // https://wikimedia.org/api/rest_v1/media/math/render/svg/5984dfb290912b0e0b92a984bf49cdd628c38b2c
    // because bctpy computes bogus
    ASSERT_NEAR(r_in_in, 0.13385826771653764, 1e-6);
    ASSERT_NEAR(r_in_out, 0.13385826771653764, 1e-6);
    ASSERT_NEAR(r_out_in, 0.13385826771653764, 1e-6);
    ASSERT_NEAR(r_out_out, 0.13385826771653764, 1e-6);
}

TEST_F(AssortativityTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    ASSERT_NEAR(r_in_in, 0.0, 1e-6);
    ASSERT_NEAR(r_in_out, 0.0, 1e-6);
    ASSERT_NEAR(r_out_in, 0.0, 1e-6);
    ASSERT_NEAR(r_out_out, 0.0, 1e-6);
}

TEST_F(AssortativityTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(r_in_in, 0.37924783647802170, 1e-6);
    ASSERT_NEAR(r_in_out, -0.11598592716925386, 1e-6);
    ASSERT_NEAR(r_out_in, 0.054735537921903650, 1e-6);
    ASSERT_NEAR(r_out_out, 0.45318711020954310, 1e-6);
}

TEST_F(AssortativityTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(r_in_in, 0.42238267148014447, 1e-6);
    ASSERT_NEAR(r_in_out, 0.42238267148014447, 1e-6);
    ASSERT_NEAR(r_out_in, 0.42238267148014447, 1e-6);
    ASSERT_NEAR(r_out_out, 0.42238267148014447, 1e-6);
}

TEST_F(AssortativityTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(r_in_in, 0.0, 1e-6);
    ASSERT_NEAR(r_in_out, 0.0, 1e-6);
    ASSERT_NEAR(r_out_in, 0.0, 1e-6);
    ASSERT_NEAR(r_out_out, 0.0, 1e-6);
}

TEST_F(AssortativityTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.37924783647802170, 1e-6);
        ASSERT_NEAR(r_in_out, -0.11598592716925386, 1e-6);
        ASSERT_NEAR(r_out_in, 0.054735537921903650, 1e-6);
        ASSERT_NEAR(r_out_out, 0.45318711020954310, 1e-6);
    }
}

TEST_F(AssortativityTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.42238267148014447, 1e-6);
        ASSERT_NEAR(r_in_out, 0.42238267148014447, 1e-6);
        ASSERT_NEAR(r_out_in, 0.42238267148014447, 1e-6);
        ASSERT_NEAR(r_out_out, 0.42238267148014447, 1e-6);
    }
}

TEST_F(AssortativityTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.0, 1e-6);
        ASSERT_NEAR(r_in_out, 0.0, 1e-6);
        ASSERT_NEAR(r_out_in, 0.0, 1e-6);
        ASSERT_NEAR(r_out_out, 0.0, 1e-6);
    }
}

TEST_F(AssortativityTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(r_in_in, 0.45983469501294240, 1e-6);
    ASSERT_NEAR(r_in_out, 0.22510459608049316, 1e-6);
    ASSERT_NEAR(r_out_in, 0.21032402550644422, 1e-6);
    ASSERT_NEAR(r_out_out, 0.64931996493649450, 1e-6);
}

TEST_F(AssortativityTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(r_in_in, 0.67689224014508020, 1e-6);
    ASSERT_NEAR(r_in_out, 0.67689224014508020, 1e-6);
    ASSERT_NEAR(r_out_in, 0.67689224014508020, 1e-6);
    ASSERT_NEAR(r_out_out, 0.67689224014508020, 1e-6);
}

TEST_F(AssortativityTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The complete graph is 62-regular, so the degrees at the two ends of an arc never vary and
    // all four correlations are reported as zero.
    // The values serve as standard for the seven-rank test
    ASSERT_NEAR(r_in_in, 0.0, 1e-6);
    ASSERT_NEAR(r_in_out, 0.0, 1e-6);
    ASSERT_NEAR(r_out_in, 0.0, 1e-6);
    ASSERT_NEAR(r_out_out, 0.0, 1e-6);
}

TEST_F(AssortativityTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.45983469501294240, 1e-6);
        ASSERT_NEAR(r_in_out, 0.22510459608049316, 1e-6);
        ASSERT_NEAR(r_out_in, 0.21032402550644422, 1e-6);
        ASSERT_NEAR(r_out_out, 0.64931996493649450, 1e-6);
    }
}

TEST_F(AssortativityTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.67689224014508020, 1e-6);
        ASSERT_NEAR(r_in_out, 0.67689224014508020, 1e-6);
        ASSERT_NEAR(r_out_in, 0.67689224014508020, 1e-6);
        ASSERT_NEAR(r_out_out, 0.67689224014508020, 1e-6);
    }
}

TEST_F(AssortativityTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.0, 1e-6);
        ASSERT_NEAR(r_in_out, 0.0, 1e-6);
        ASSERT_NEAR(r_out_in, 0.0, 1e-6);
        ASSERT_NEAR(r_out_out, 0.0, 1e-6);
    }
}
