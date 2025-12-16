#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIReductions.h"

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

/**
 * @brief This class offers a common interface to report the status of a computation.
 *		The results are reported on MPI rank 0.
 */
class Status {
public:
    /**
     * @brief Constructs the object. Must be called on all MPI ranks.
     * @param number_iterations The number of local iterations
     * @param algorithm_name The name of the algorithm
     */
    Status(const std::uint64_t number_iterations, std::string algorithm_name)
        : algo_name(std::move(algorithm_name))
        , num_local_iterations(number_iterations)
        , min_num_local_iterations(mpiPP::MPIReductions::all_reduce_min(number_iterations))
        , num_global_iterations(mpiPP::MPIReductions::reduce_sum(number_iterations)) {
    }

    /**
     * @brief Reports the current progress. Must be called on all MPI ranks.
     * @param current_iteration The number of completed local iterations
     */
    void report([[maybe_unused]] const std::uint64_t current_iteration) {
        if (disable_status) {
            return;
        }

        num_reported_iterations++;

        if (const auto print_iterations = num_local_iterations / 100; print_iterations != 0 && num_reported_iterations % print_iterations != 0) {
            return;
        }

        ++num_printed_iterations;
        if (num_printed_iterations > 101 || num_printed_iterations > min_num_local_iterations) {
            return;
        }

        const auto total_iterations = mpiPP::MPIReductions::reduce_sum(num_reported_iterations);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return;
        }

        fmt::print("{}: Processed a total of {} of {} iterations.\n", algo_name, total_iterations, num_global_iterations);
    }

    /**
     * @brief Finishes the status reporting. Prints to std::cout on MPI rank 0.
     */
    void finish() const {
        if (disable_status) {
            return;
        }

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return;
        }

        fmt::print("Processed all iterations of {}\n", algo_name);
    }

    /**
     * @brief Sets the disabled status.
     * @param status True iff the status reporting should be disabled
     */
    static void set_disable_status(const bool status) {
        disable_status = status;
    }

private:
    std::string algo_name{};
    std::uint64_t num_reported_iterations{};
    std::uint64_t num_local_iterations{};
    std::uint64_t min_num_local_iterations{};
    std::uint64_t num_global_iterations{};
    std::uint64_t num_printed_iterations{};

    static inline bool disable_status{};
};
