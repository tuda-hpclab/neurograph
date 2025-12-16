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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief This class offers a common interface to report the time of a computation.
 *		The results are reported on MPI rank 0.
 */
class Timer {
    using hrc = std::chrono::high_resolution_clock;
    using tp = std::chrono::time_point<hrc>;

public:
    /**
     * @brief Constructs the object
     * @param algorithm_name The name of the algorithm
     * @param reserve The number of timings that are expected to happen
     */
    explicit Timer(std::string algorithm_name, const std::size_t reserve = 20)
        : algo_name(std::move(algorithm_name))
        , start_point(hrc::now()) {
        time_points.reserve(reserve);
    }

    /**
     * @brief Times the current section; should be called after its completion.
     * @param name The name of the current section
     */
    void time_section(std::string name) {
        if (disable_timer) {
            return;
        }

        time_points.emplace_back(std::move(name), hrc::now());
    }

    /**
     * @brief Finishes the timing. Must be called on every MPI rank.
     *		Prints the times to std::cout on MPI rank 0.
     */
    void finish() {
        if (disable_timer) {
            return;
        }

        if (time_points.empty()) {
            return;
        }

        using gcc_is_an_idiot_type = long long;

        const auto number_time_points = time_points.size();

        auto times = std::vector<gcc_is_an_idiot_type>(number_time_points, 0);
        if (times.empty()) {
            // This exists because some GCCs think times could be empty and issue warnings based on that
            return;
        }

        times[0] = std::chrono::duration_cast<std::chrono::milliseconds>(time_points[0].second - start_point).count();

        for (auto it = std::size_t{ 1 }; it < number_time_points; ++it) {
            times[it] = std::chrono::duration_cast<std::chrono::milliseconds>(time_points[it].second - time_points[it - 1].second).count();
        }

        auto global_avg_times = mpiPP::MPIReductions::reduce_componentwise_sum(times);
        auto global_max_times = mpiPP::MPIReductions::reduce_componentwise_max(times);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return;
        }

        std::ranges::for_each(global_avg_times, [](gcc_is_an_idiot_type& time_value) { time_value /= mpiPP::MPIInfo::get_number_ranks(); });

        fmt::print("{}\n", algo_name);

        for (auto it = std::size_t{ 0 }; it < time_points.size(); ++it) {
            const auto average = global_avg_times[it];
            const auto maximum = global_max_times[it];

            const auto& [time_name, _] = time_points[it];

            fmt::print("{:<25}:\tavg: {:>6}\tmax: {:>6}\tmilliseconds\n", time_name, average, maximum);
        }

        const auto total_avg = std::accumulate(global_avg_times.begin(), global_avg_times.end(), gcc_is_an_idiot_type{ 0 });
        const auto total_max = std::accumulate(global_max_times.begin(), global_max_times.end(), gcc_is_an_idiot_type{ 0 });

        fmt::print("{:<25}:\tavg: {:>6}\tmax: {:>6}\tmilliseconds\n----------------------------------\n", "Total", total_avg, total_max);
    }

    /**
     * @brief Sets the disabled status.
     * @param status True iff the timing should be disabled
     */
    static void set_disable_status(const bool status) {
        disable_timer = status;
    }

private:
    std::string algo_name{};
    tp start_point{};
    std::vector<std::pair<std::string, tp>> time_points{};

    static inline bool disable_timer{};
};
