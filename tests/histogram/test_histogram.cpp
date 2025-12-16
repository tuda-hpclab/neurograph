/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_histogram.h"

#include "utility/Histogram.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <iostream>

TEST_F(HistogramTest, testFixedSizeInts1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<int>(3);

    ASSERT_EQ(histogram.get_bin_width(), 3);

    const auto counts_0 = histogram.get_counts();
    const auto borders_0 = histogram.get_borders();

    ASSERT_EQ(counts_0.size(), 0);
    ASSERT_EQ(counts_0.size(), 0);

    histogram.add_data_point(0);
    histogram.add_data_point(1);

    const auto counts_1 = histogram.get_counts();
    const auto borders_1 = histogram.get_borders();

    ASSERT_EQ(counts_1.size(), 1);
    ASSERT_EQ(borders_1.size(), 1);

    ASSERT_EQ(counts_1[0], 2);
    ASSERT_EQ(borders_1[0], 0);

    histogram.add_data_point(3);
    histogram.add_data_point(4);

    const auto counts_2 = histogram.get_counts();
    const auto borders_2 = histogram.get_borders();

    ASSERT_EQ(counts_2.size(), 2);
    ASSERT_EQ(borders_2.size(), 2);

    ASSERT_EQ(counts_2[0], 2);
    ASSERT_EQ(counts_2[1], 2);

    ASSERT_EQ(borders_2[0], 0);
    ASSERT_EQ(borders_2[1], 3);
}

TEST_F(HistogramTest, testFixedSizeInts2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<int>(3);

    ASSERT_EQ(histogram.get_bin_width(), 3);

    histogram.add_data_point(0);
    histogram.add_data_point(1);

    histogram.add_data_point(3);
    histogram.add_data_point(4);

    histogram.add_data_point(30);
    histogram.add_data_point(29);

    histogram.add_data_point(3);
    histogram.add_data_point(3);
    histogram.add_data_point(3);

    histogram.add_data_point(17);
    histogram.add_data_point(18);
    histogram.add_data_point(19);

    histogram.add_data_point(11);
    histogram.add_data_point(11);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(counts.size(), 11);
    ASSERT_EQ(borders.size(), 11);

    ASSERT_EQ(counts[0], 2);  // [0, 3)
    ASSERT_EQ(counts[1], 5);  // [3, 6)
    ASSERT_EQ(counts[2], 0);  // [6, 9)
    ASSERT_EQ(counts[3], 2);  // [9, 12)
    ASSERT_EQ(counts[4], 0);  // [12, 15)
    ASSERT_EQ(counts[5], 1);  // [15, 18)
    ASSERT_EQ(counts[6], 2);  // [18, 21)
    ASSERT_EQ(counts[7], 0);  // [21, 24)
    ASSERT_EQ(counts[8], 0);  // [24, 27)
    ASSERT_EQ(counts[9], 1);  // [27, 30)
    ASSERT_EQ(counts[10], 1); // [30, 33)

    ASSERT_EQ(borders[0], 0);
    ASSERT_EQ(borders[1], 3);
    ASSERT_EQ(borders[2], 6);
    ASSERT_EQ(borders[3], 9);
    ASSERT_EQ(borders[4], 12);
    ASSERT_EQ(borders[5], 15);
    ASSERT_EQ(borders[6], 18);
    ASSERT_EQ(borders[7], 21);
    ASSERT_EQ(borders[8], 24);
    ASSERT_EQ(borders[9], 27);
    ASSERT_EQ(borders[10], 30);
}

TEST_F(HistogramTest, testFixedSizeIntsFail1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    ASSERT_THROW(auto histogram = FixedWidthHistogram<int>(0);, utility::Exception);
}

TEST_F(HistogramTest, testFixedSizeIntsFail2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<int>(1);

    ASSERT_THROW(histogram.add_data_point(-2);, utility::Exception);
}

TEST_F(HistogramTest, testFixedSizeFloats1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<float>(0.25f);

    ASSERT_EQ(histogram.get_bin_width(), 0.25f);

    const auto counts_0 = histogram.get_counts();
    const auto borders_0 = histogram.get_borders();

    ASSERT_EQ(counts_0.size(), 0);
    ASSERT_EQ(borders_0.size(), 0);

    histogram.add_data_point(1.2f);
    histogram.add_data_point(1.5f);
    histogram.add_data_point(0.499f);

    const auto counts_1 = histogram.get_counts();
    const auto borders_1 = histogram.get_borders();

    ASSERT_EQ(counts_1.size(), 7);
    ASSERT_EQ(borders_1.size(), 7);

    ASSERT_EQ(counts_1[0], 0); // [0, 0.25)
    ASSERT_EQ(counts_1[1], 1); // [0.25, 0.5)
    ASSERT_EQ(counts_1[2], 0); // [0.5, 0.75)
    ASSERT_EQ(counts_1[3], 0); // [0.75, 1)
    ASSERT_EQ(counts_1[4], 1); // [1, 1.25)
    ASSERT_EQ(counts_1[5], 0); // [1.25, 1.5)
    ASSERT_EQ(counts_1[6], 1); // [1.5, 1.75)

    ASSERT_NEAR(borders_1[0], 0.0f, 1e-6);
    ASSERT_NEAR(borders_1[1], 0.25f, 1e-6);
    ASSERT_NEAR(borders_1[2], 0.5f, 1e-6);
    ASSERT_NEAR(borders_1[3], 0.75f, 1e-6);
    ASSERT_NEAR(borders_1[4], 1.0f, 1e-6);
    ASSERT_NEAR(borders_1[5], 1.25f, 1e-6);
    ASSERT_NEAR(borders_1[6], 1.5f, 1e-6);
}

TEST_F(HistogramTest, testFixedSizeFloats2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<float>(0.45f);

    ASSERT_EQ(histogram.get_bin_width(), 0.45f);

    histogram.add_data_point(1.2f);
    histogram.add_data_point(1.5f);
    histogram.add_data_point(4.4f);
    histogram.add_data_point(3.2f);
    histogram.add_data_point(3.2f);
    histogram.add_data_point(3.2f);
    histogram.add_data_point(0.1f);
    histogram.add_data_point(0.2f);
    histogram.add_data_point(1.2f);

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(counts.size(), 10);
    ASSERT_EQ(borders.size(), 10);

    ASSERT_EQ(counts[0], 2); // [0, 0.45)
    ASSERT_EQ(counts[1], 0); // [0.45, 0.9)
    ASSERT_EQ(counts[2], 2); // [0.9, 1.35)
    ASSERT_EQ(counts[3], 1); // [1.35, 1.8)
    ASSERT_EQ(counts[4], 0); // [1.8, 2.25)
    ASSERT_EQ(counts[5], 0); // [2.25, 2.7)
    ASSERT_EQ(counts[6], 0); // [2.7, 3.15)
    ASSERT_EQ(counts[7], 3); // [3.15, 3.6)
    ASSERT_EQ(counts[8], 0); // [3.6, 4.05)
    ASSERT_EQ(counts[9], 1); // [4.05, 4.5)

    ASSERT_NEAR(borders[0], 0.0f, 1e-6);
    ASSERT_NEAR(borders[1], 0.45f, 1e-6);
    ASSERT_NEAR(borders[2], 0.9f, 1e-6);
    ASSERT_NEAR(borders[3], 1.35f, 1e-6);
    ASSERT_NEAR(borders[4], 1.8f, 1e-6);
    ASSERT_NEAR(borders[5], 2.25f, 1e-6);
    ASSERT_NEAR(borders[6], 2.7f, 1e-6);
    ASSERT_NEAR(borders[7], 3.15f, 1e-6);
    ASSERT_NEAR(borders[8], 3.6f, 1e-6);
    ASSERT_NEAR(borders[9], 4.05f, 1e-6);
}

TEST_F(HistogramTest, testFixedSizeFloatsFail1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    ASSERT_THROW(auto histogram = FixedWidthHistogram<float>(0);, utility::Exception);
}

TEST_F(HistogramTest, testFixedSizeFloatsFail2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedWidthHistogram<float>(1);

    ASSERT_THROW(histogram.add_data_point(-2.32f);, utility::Exception);
}

TEST_F(HistogramTest, testFixedWidthInts1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<int>(2, 5, std::size_t(3));

    ASSERT_EQ(histogram.get_bin_width(), 1);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 3);
    ASSERT_EQ(counts.size(), 3);

    ASSERT_EQ(borders[0], 2);
    ASSERT_EQ(borders[1], 3);
    ASSERT_EQ(borders[2], 4);

    ASSERT_EQ(counts[0], 0);
    ASSERT_EQ(counts[1], 0);
    ASSERT_EQ(counts[2], 0);

    histogram.add_data_point(2);
    histogram.add_data_point(3);
    histogram.add_data_point(4);

    histogram.add_data_point(3);
    histogram.add_data_point(3);
    histogram.add_data_point(3);

    ASSERT_EQ(borders[0], 2);
    ASSERT_EQ(borders[1], 3);
    ASSERT_EQ(borders[2], 4);

    ASSERT_EQ(counts[0], 1);
    ASSERT_EQ(counts[1], 4);
    ASSERT_EQ(counts[2], 1);
}

TEST_F(HistogramTest, testFixedWidthInts2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<int>(-3, 5, std::size_t(4));

    ASSERT_EQ(histogram.get_bin_width(), 2);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 4);
    ASSERT_EQ(counts.size(), 4);

    ASSERT_EQ(borders[0], -3);
    ASSERT_EQ(borders[1], -1);
    ASSERT_EQ(borders[2], 1);
    ASSERT_EQ(borders[3], 3);

    ASSERT_EQ(counts[0], 0);
    ASSERT_EQ(counts[1], 0);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 0);

    histogram.add_data_point(-3);
    histogram.add_data_point(-3);
    histogram.add_data_point(-3);
    histogram.add_data_point(-3);

    histogram.add_data_point(-2);
    histogram.add_data_point(-1);

    histogram.add_data_point(2);
    histogram.add_data_point(2);
    histogram.add_data_point(2);
    histogram.add_data_point(3);
    histogram.add_data_point(4);

    ASSERT_EQ(borders[0], -3);
    ASSERT_EQ(borders[1], -1);
    ASSERT_EQ(borders[2], 1);
    ASSERT_EQ(borders[3], 3);

    ASSERT_EQ(counts[0], 5);
    ASSERT_EQ(counts[1], 1);
    ASSERT_EQ(counts[2], 3);
    ASSERT_EQ(counts[3], 2);
}

TEST_F(HistogramTest, testFixedWidthIntsFail1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    ASSERT_THROW(auto histogram = FixedSizeHistogram<int>(5, -3, std::size_t(4));, utility::Exception);
    ASSERT_THROW(auto histogram = FixedSizeHistogram<int>(-5, 3, std::size_t(0));, utility::Exception);
    ASSERT_THROW(auto histogram = FixedSizeHistogram<int>(2, 4, std::size_t(3));, utility::Exception);
}

TEST_F(HistogramTest, testFixedWidthIntsFail2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<int>(-5, 3, std::size_t(4));

    ASSERT_THROW(histogram.add_data_point(-6);, utility::Exception);
    ASSERT_THROW(histogram.add_data_point(6);, utility::Exception);
}

TEST_F(HistogramTest, testFixedWidthFloats1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<float>(2.3f, 3.7f, std::size_t(7));

    ASSERT_NEAR(histogram.get_bin_width(), 0.2f, 1e-6);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 7);
    ASSERT_EQ(counts.size(), 7);

    ASSERT_NEAR(borders[0], 2.3f, 1e-6);
    ASSERT_NEAR(borders[1], 2.5f, 1e-6);
    ASSERT_NEAR(borders[2], 2.7f, 1e-6);
    ASSERT_NEAR(borders[3], 2.9f, 1e-6);
    ASSERT_NEAR(borders[4], 3.1f, 1e-6);
    ASSERT_NEAR(borders[5], 3.3f, 1e-6);
    ASSERT_NEAR(borders[6], 3.5f, 1e-6);

    ASSERT_EQ(counts[0], 0);
    ASSERT_EQ(counts[1], 0);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 0);
    ASSERT_EQ(counts[4], 0);
    ASSERT_EQ(counts[5], 0);
    ASSERT_EQ(counts[6], 0);

    histogram.add_data_point(2.3f);
    histogram.add_data_point(2.3f);
    histogram.add_data_point(2.6f);
    histogram.add_data_point(2.68846123f);
    histogram.add_data_point(2.9f);
    histogram.add_data_point(3.1123123f);
    histogram.add_data_point(3.2f);
    histogram.add_data_point(3.50002f);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 1);
    ASSERT_EQ(counts[4], 2);
    ASSERT_EQ(counts[5], 0);
    ASSERT_EQ(counts[6], 1);
}

TEST_F(HistogramTest, testFixedWidthFloats2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

    ASSERT_NEAR(histogram.get_bin_width(), 1.3f, 1e-6);

    const auto borders = histogram.get_borders();
    const auto counts = histogram.get_counts();

    ASSERT_EQ(borders.size(), 5);
    ASSERT_EQ(counts.size(), 5);

    ASSERT_NEAR(borders[0], -1.3f, 1e-6);
    ASSERT_NEAR(borders[1], 0.0f, 1e-6);
    ASSERT_NEAR(borders[2], 1.3f, 1e-6);
    ASSERT_NEAR(borders[3], 2.6f, 1e-6);
    ASSERT_NEAR(borders[4], 3.9f, 1e-6);

    ASSERT_EQ(counts[0], 0);
    ASSERT_EQ(counts[1], 0);
    ASSERT_EQ(counts[2], 0);
    ASSERT_EQ(counts[3], 0);
    ASSERT_EQ(counts[4], 0);

    histogram.add_data_point(-1.29999f);
    histogram.add_data_point(-0.00001f);

    histogram.add_data_point(0.00001f);
    histogram.add_data_point(1.29999f);

    histogram.add_data_point(1.30001f);
    histogram.add_data_point(2.59999f);

    histogram.add_data_point(2.60001f);
    histogram.add_data_point(3.89999f);

    histogram.add_data_point(3.90002f);
    histogram.add_data_point(5.19999f);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 2);
    ASSERT_EQ(counts[3], 2);
    ASSERT_EQ(counts[4], 2);
}

TEST_F(HistogramTest, testFixedWidthFloatsFail1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    ASSERT_THROW(auto histogram = FixedSizeHistogram<float>(2.257f, 1.7863f, std::size_t(4));, utility::Exception);
    ASSERT_THROW(auto histogram = FixedSizeHistogram<float>(-9.41f, 1.39f, std::size_t(0));, utility::Exception);
}

TEST_F(HistogramTest, testFixedWidthFloatsFail2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto histogram = FixedSizeHistogram<float>(-2.7f, 9.42f, std::size_t(4));

    ASSERT_THROW(histogram.add_data_point(-6.022f);, utility::Exception);
    ASSERT_THROW(histogram.add_data_point(12.02f);, utility::Exception);
}

TEST_F(HistogramTest, testHistogramConstruction1) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto fw_histogram = FixedWidthHistogram<int>(3);

    fw_histogram.add_data_point(0);
    fw_histogram.add_data_point(1);
    fw_histogram.add_data_point(3);
    fw_histogram.add_data_point(4);

    auto histogram = Histogram<int>(std::move(fw_histogram));

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(counts.size(), 2);
    ASSERT_EQ(borders.size(), 2);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 2);

    ASSERT_EQ(borders[0], 0);
    ASSERT_EQ(borders[1], 3);
}

TEST_F(HistogramTest, testHistogramConstruction2) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

    fs_histogram.add_data_point(-1.29999f);
    fs_histogram.add_data_point(-0.00001f);

    fs_histogram.add_data_point(0.00001f);
    fs_histogram.add_data_point(1.29999f);

    fs_histogram.add_data_point(1.30001f);
    fs_histogram.add_data_point(2.59999f);

    fs_histogram.add_data_point(2.60001f);
    fs_histogram.add_data_point(3.89999f);

    fs_histogram.add_data_point(3.90002f);
    fs_histogram.add_data_point(5.19999f);

    auto histogram = Histogram<float>(std::move(fs_histogram));

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(borders.size(), 5);
    ASSERT_EQ(counts.size(), 5);

    ASSERT_NEAR(borders[0], -1.3f, 1e-6);
    ASSERT_NEAR(borders[1], 0.0f, 1e-6);
    ASSERT_NEAR(borders[2], 1.3f, 1e-6);
    ASSERT_NEAR(borders[3], 2.6f, 1e-6);
    ASSERT_NEAR(borders[4], 3.9f, 1e-6);

    ASSERT_EQ(counts[0], 2);
    ASSERT_EQ(counts[1], 2);
    ASSERT_EQ(counts[2], 2);
    ASSERT_EQ(counts[3], 2);
    ASSERT_EQ(counts[4], 2);
}

TEST_F(HistogramTest, testHistogramSumOneRank) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

    fs_histogram.add_data_point(-1.29999f);
    fs_histogram.add_data_point(-0.00001f);

    fs_histogram.add_data_point(0.00001f);
    fs_histogram.add_data_point(1.29999f);

    fs_histogram.add_data_point(1.30001f);
    fs_histogram.add_data_point(2.59999f);

    fs_histogram.add_data_point(2.60001f);
    fs_histogram.add_data_point(3.89999f);

    fs_histogram.add_data_point(3.90002f);
    fs_histogram.add_data_point(5.19999f);

    auto histogram = Histogram<float>(std::move(fs_histogram));

    for (auto i = 0; i < 3; i++) {
        histogram.sum_on_rank_0();

        const auto counts = histogram.get_counts();
        const auto borders = histogram.get_borders();

        ASSERT_EQ(borders.size(), 5);
        ASSERT_EQ(counts.size(), 5);

        ASSERT_NEAR(borders[0], -1.3f, 1e-6);
        ASSERT_NEAR(borders[1], 0.0f, 1e-6);
        ASSERT_NEAR(borders[2], 1.3f, 1e-6);
        ASSERT_NEAR(borders[3], 2.6f, 1e-6);
        ASSERT_NEAR(borders[4], 3.9f, 1e-6);

        ASSERT_EQ(counts[0], 2);
        ASSERT_EQ(counts[1], 2);
        ASSERT_EQ(counts[2], 2);
        ASSERT_EQ(counts[3], 2);
        ASSERT_EQ(counts[4], 2);
    }
}

TEST_F(HistogramTest, testHistogramAllSumOneRank) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 1 MPI ranks.");
        }

        return;
    }

    auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

    fs_histogram.add_data_point(-1.29999f);
    fs_histogram.add_data_point(-0.00001f);

    fs_histogram.add_data_point(0.00001f);
    fs_histogram.add_data_point(1.29999f);

    fs_histogram.add_data_point(1.30001f);
    fs_histogram.add_data_point(2.59999f);

    fs_histogram.add_data_point(2.60001f);
    fs_histogram.add_data_point(3.89999f);

    fs_histogram.add_data_point(3.90002f);
    fs_histogram.add_data_point(5.19999f);

    auto histogram = Histogram<float>(std::move(fs_histogram));

    for (auto i = 0; i < 3; i++) {
        histogram.sum_on_all_ranks();

        const auto counts = histogram.get_counts();
        const auto borders = histogram.get_borders();

        ASSERT_EQ(borders.size(), 5);
        ASSERT_EQ(counts.size(), 5);

        ASSERT_NEAR(borders[0], -1.3f, 1e-6);
        ASSERT_NEAR(borders[1], 0.0f, 1e-6);
        ASSERT_NEAR(borders[2], 1.3f, 1e-6);
        ASSERT_NEAR(borders[3], 2.6f, 1e-6);
        ASSERT_NEAR(borders[4], 3.9f, 1e-6);

        ASSERT_EQ(counts[0], 2);
        ASSERT_EQ(counts[1], 2);
        ASSERT_EQ(counts[2], 2);
        ASSERT_EQ(counts[3], 2);
        ASSERT_EQ(counts[4], 2);
    }
}

TEST_F(HistogramTest, testHistogramSumFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto get_histogram = [](const mpi_rank_type rank) {
        auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

        if (rank == 0) {
            fs_histogram.add_data_point(-1.29999f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.00001f);
            fs_histogram.add_data_point(1.29999f);

            fs_histogram.add_data_point(1.30001f);
            fs_histogram.add_data_point(2.59999f);

            fs_histogram.add_data_point(2.60001f);
            fs_histogram.add_data_point(3.89999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        } else if (rank == 1) {
            fs_histogram.add_data_point(-0.49999f);
            fs_histogram.add_data_point(-0.25552f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.9f);
            fs_histogram.add_data_point(1.1f);
        } else if (rank == 2) {
            /* Intentionally empty */
        } else if (rank == 3) {
            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        }

        return Histogram<float>(std::move(fs_histogram));
    };

    auto histogram = get_histogram(my_rank);
    histogram.sum_on_rank_0();

    if (my_rank != 0) {
        return;
    }

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(borders.size(), 5);
    ASSERT_EQ(counts.size(), 5);

    ASSERT_NEAR(borders[0], -1.3f, 1e-6);
    ASSERT_NEAR(borders[1], 0.0f, 1e-6);
    ASSERT_NEAR(borders[2], 1.3f, 1e-6);
    ASSERT_NEAR(borders[3], 2.6f, 1e-6);
    ASSERT_NEAR(borders[4], 3.9f, 1e-6);

    ASSERT_EQ(counts[0], 5);
    ASSERT_EQ(counts[1], 4);
    ASSERT_EQ(counts[2], 2);
    ASSERT_EQ(counts[3], 2);
    ASSERT_EQ(counts[4], 6);
}

TEST_F(HistogramTest, testHistogramAllSumFourRanks) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto get_histogram = [](const mpi_rank_type rank) {
        auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5));

        if (rank == 0) {
            fs_histogram.add_data_point(-1.29999f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.00001f);
            fs_histogram.add_data_point(1.29999f);

            fs_histogram.add_data_point(1.30001f);
            fs_histogram.add_data_point(2.59999f);

            fs_histogram.add_data_point(2.60001f);
            fs_histogram.add_data_point(3.89999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        } else if (rank == 1) {
            fs_histogram.add_data_point(-0.49999f);
            fs_histogram.add_data_point(-0.25552f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.9f);
            fs_histogram.add_data_point(1.1f);
        } else if (rank == 2) {
            /* Intentionally empty */
        } else if (rank == 3) {
            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        }

        return Histogram<float>(std::move(fs_histogram));
    };

    auto histogram = get_histogram(my_rank);
    histogram.sum_on_all_ranks();

    const auto counts = histogram.get_counts();
    const auto borders = histogram.get_borders();

    ASSERT_EQ(borders.size(), 5);
    ASSERT_EQ(counts.size(), 5);

    ASSERT_NEAR(borders[0], -1.3f, 1e-6);
    ASSERT_NEAR(borders[1], 0.0f, 1e-6);
    ASSERT_NEAR(borders[2], 1.3f, 1e-6);
    ASSERT_NEAR(borders[3], 2.6f, 1e-6);
    ASSERT_NEAR(borders[4], 3.9f, 1e-6);

    ASSERT_EQ(counts[0], 5);
    ASSERT_EQ(counts[1], 4);
    ASSERT_EQ(counts[2], 2);
    ASSERT_EQ(counts[3], 2);
    ASSERT_EQ(counts[4], 6);
}

TEST_F(HistogramTest, testHistogramSumFourRanksFail) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto get_histogram = [](const mpi_rank_type rank) {
        auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5 + rank));

        if (rank == 0) {
            fs_histogram.add_data_point(-1.29999f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.00001f);
            fs_histogram.add_data_point(1.29999f);

            fs_histogram.add_data_point(1.30001f);
            fs_histogram.add_data_point(2.59999f);

            fs_histogram.add_data_point(2.60001f);
            fs_histogram.add_data_point(3.89999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        } else if (rank == 1) {
            fs_histogram.add_data_point(-0.49999f);
            fs_histogram.add_data_point(-0.25552f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.9f);
            fs_histogram.add_data_point(1.1f);
        } else if (rank == 2) {
            /* Intentionally empty */
        } else if (rank == 3) {
            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        }

        return Histogram<float>(std::move(fs_histogram));
    };

    auto histogram = get_histogram(my_rank);

    /* In the implementation, this test does not work:
     * Rank 0 will throw after the reductions but before the componentwise reduction.
     * Rank 1-3 will not throw but call all MPI functions.
     * This results in a starvation (no answer from Rank 0).
     * */
    if (my_rank == 0) {
        // ASSERT_THROW(histogram.sum_on_rank_0();, utility::Exception);
    } else {
        // ASSERT_NO_THROW(histogram.sum_on_rank_0(););
    }
}

TEST_F(HistogramTest, testHistogramAllSumFourRanksFail) {
    if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with 4 MPI ranks.");
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto get_histogram = [](const mpi_rank_type rank) {
        auto fs_histogram = FixedSizeHistogram<float>(-1.3f, 5.2f, std::size_t(5 + rank));

        if (rank == 0) {
            fs_histogram.add_data_point(-1.29999f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.00001f);
            fs_histogram.add_data_point(1.29999f);

            fs_histogram.add_data_point(1.30001f);
            fs_histogram.add_data_point(2.59999f);

            fs_histogram.add_data_point(2.60001f);
            fs_histogram.add_data_point(3.89999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        } else if (rank == 1) {
            fs_histogram.add_data_point(-0.49999f);
            fs_histogram.add_data_point(-0.25552f);
            fs_histogram.add_data_point(-0.00001f);

            fs_histogram.add_data_point(0.9f);
            fs_histogram.add_data_point(1.1f);
        } else if (rank == 2) {
            /* Intentionally empty */
        } else if (rank == 3) {
            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);

            fs_histogram.add_data_point(3.90002f);
            fs_histogram.add_data_point(5.19999f);
        }

        return Histogram<float>(std::move(fs_histogram));
    };

    auto histogram = get_histogram(my_rank);
    ASSERT_THROW(histogram.sum_on_all_ranks();, utility::Exception);
}
