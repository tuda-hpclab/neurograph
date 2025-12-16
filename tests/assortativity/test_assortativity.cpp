/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_assortativity.h"

#include "metrics/Assortativity.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(AssortativityTest, testStandard) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    // The values serve as standard for the four-rank test
    ASSERT_NEAR(r_in_in, 0.40232684829609511, 1e-6);
    ASSERT_NEAR(r_in_out, -0.085489081561593488, 1e-6);
    ASSERT_NEAR(r_out_in, 0.11325082425467951, 1e-6);
    ASSERT_NEAR(r_out_out, 0.44805633478760648, 1e-6);
}

TEST_F(AssortativityTest, testStandardUUFourRanksDummy) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto assortativity = Assortativity::compute_assortativity(graph);
    const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity;

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The values are the corresponding ones from the one-rank version
        ASSERT_NEAR(r_in_in, 0.40232684829609511, 1e-6);
        ASSERT_NEAR(r_in_out, -0.085489081561593488, 1e-6);
        ASSERT_NEAR(r_out_in, 0.11325082425467951, 1e-6);
        ASSERT_NEAR(r_out_out, 0.44805633478760648, 1e-6);
    }
}

TEST_F(AssortativityTest, testStandardUUFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

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
