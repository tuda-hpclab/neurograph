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

#include "metrics/connectivity/SccVariants/DistributedSccState.h"
#include "metrics/connectivity/SccVariants/SccCommon.h"

/**
 * @brief Strongly connected components via the forward-backward decomposition by Fleischer,
 *		Hendrickson, and Pinar (2000), "On identifying strongly connected components in parallel", on
 *		the distributed graph. Each round picks a pivot and peels off the nodes that the pivot reaches
 *		and that reach the pivot, which is exactly one component. Unlike the original recursive
 *		formulation, the rounds stay flat: all ranks work on the whole remaining graph together instead
 *		of splitting into independent subproblems, which needs neither sub-communicators nor a task
 *		pool and keeps the code in the shape of the other distributed phases.
 *		One round costs two distributed traversals, so this variant is the fastest of the distributed
 *		ones on a graph with few components, but it needs one round per component and is therefore a
 *		poor choice for the many tiny components that real graphs have; MultistepVariant uses a single
 *		round of it to peel the one large component and leaves the rest to the coloring rounds.
 */
class ForwardBackwardVariant {
public:
    /**
     * @brief Determines the strongly connected components of the whole graph.
     *		Must be called on every MPI rank, returns the same result on every MPI rank.
     * @param state The state, which is fully processed afterwards
     * @return The components, summarized by their sizes
     */
    [[nodiscard]] static SccResult all_compute_components(DistributedSccState& state) {
        while (true) {
            const auto pivot = state.all_select_pivot();

            // Every round removes at least the pivot, so the rounds run out
            if (pivot.has_uninitialized_rank()) {
                break;
            }

            state.all_assign_component_of(pivot);
        }

        return SccCommon::all_assemble_result(state.all_count_component_sizes());
    }
};
