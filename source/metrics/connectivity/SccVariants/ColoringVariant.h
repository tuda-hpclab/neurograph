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
 * @brief Strongly connected components via the coloring method by Orzan (2004), "On Distributed
 *		Verification and Verified Distribution", on the distributed graph. Each round labels the
 *		remaining nodes with the largest global id that reaches them and peels off the component of
 *		every node that kept its own label, so one round finds all components that no larger id reaches
 *		instead of a single one.
 *		That makes it the better distributed choice whenever a graph consists of many small components,
 *		which is where ForwardBackwardVariant needs one round per component; on a graph that is one
 *		large component it is slower, because a color has to travel through the whole component before
 *		anything is peeled off.
 */
class ColoringVariant {
public:
    /**
     * @brief Determines the strongly connected components of the whole graph.
     *		Must be called on every MPI rank, returns the same result on every MPI rank.
     * @param state The state, which is fully processed afterwards
     * @return The components, summarized by their sizes
     */
    [[nodiscard]] static SccResult all_compute_components(DistributedSccState& state) {
        // Every round removes at least the component of the largest remaining global id
        while (state.all_count_active_nodes() > 0) {
            state.all_assign_components_by_coloring();
        }

        return SccCommon::all_assemble_result(state.all_count_component_sizes());
    }
};
