#pragma once

/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "metrics/paths/DeltaSteppingVariants/Buckets.h"

#include <mpi-wrapper/reductions/MPIReductions.h>

#include <cstddef>

/**
 * @brief Shared helpers used by every delta-stepping epoch variant (see the sibling headers in
 *		metrics/paths/DeltaSteppingVariants).
 */
class EpochCommon {
public:
    /**
     * @brief Checks if the ranks need another epoch
     * @param buckets The currently used buckets
     * @param planned_epoch The index of the next epoch
     * @return True iff the algorithm needs another epoch
     */
    [[nodiscard]] static bool check_other_epoch_is_necessary(const Buckets& buckets, const std::size_t planned_epoch) {
        const auto locally_need_another_epoch = buckets.has_nodes_left(planned_epoch);
        const auto globally_need_another_epoch = mpiPP::MPIReductions::all_reduce_or(locally_need_another_epoch);

        return globally_need_another_epoch;
    }

    /**
     * @brief Calculates the index of the bucket for a node with the given distance
     * @param delta The current delta
     * @param current_distance The given distance of a node
     * @return The index of the bucket
     */
    [[nodiscard]] static std::size_t compute_bucket_index(const distance_type delta, const distance_type current_distance) {
        const auto index = current_distance / delta;
        return index;
    }
};
