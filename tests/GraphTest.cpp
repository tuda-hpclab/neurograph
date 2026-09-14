/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GraphTest.h"

#include <mpi-wrapper/MPIWrapper.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>

#include <spdlog/spdlog.h>

bool GraphTest::skip_unless_rank_count(const int required_number_ranks) {
    if (mpiPP::MPIInfo::get_number_ranks() != required_number_ranks) {
        if (mpiPP::MPIInfo::is_root_rank()) {
            spdlog::info("Test only works with {} MPI ranks.", required_number_ranks);
        }

        return true;
    }

    return false;
}

void GraphTest::SetUp() {
    // Keep the ranks from drifting apart over the course of the test suite: each test starts with all ranks at the same point
    mpiPP::MPISynchronization::barrier();
}

void GraphTest::TearDown() {
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    mpiPP::MPIWrapper::init(argc, argv);

    // The metrics report their progress and their region timings on the root rank, which buries the output of a
    // failing test under hundreds of lines. Both switches only suppress the printing: the progress reporter
    // communicates in its constructor alone, and the timer report reduces even while its printing is disabled,
    // so the timers stay measured (and covered) and no rank skips a collective.
    mpiPP::MPIProgress::set_disable_status(true);
    mpiPP::MPIRegionTimerReport::set_disable_status(true);

    if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
        // std::this_thread::sleep_for(std::chrono::seconds{30});
    }

    const auto tests_return_code = RUN_ALL_TESTS();

    mpiPP::MPIWrapper::finalize();
    return tests_return_code;
}
