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
 * @brief Strongly connected components via Tarjan's algorithm (1972), "Depth-first search and linear
 *		graph algorithms", on the root rank. A depth-first search visits the nodes in an order that no
 *		data decomposition can reproduce, so this variant gathers the arcs on the root rank and computes
 *		the components there in O(V + A), which is optimal but limited by the memory of a single rank.
 *		It therefore serves as the reference the distributed variants are compared against, and as the
 *		last phase of MultistepVariant once the unprocessed part of the graph is small enough.
 */
class TarjanGatheredVariant {
public:
    /**
     * @brief Determines the strongly connected components of the whole graph.
     *		Must be called on every MPI rank, returns the same result on every MPI rank.
     * @param state The state, which must be untouched so that the whole graph is still unprocessed
     * @return The components, summarized by their sizes
     */
    [[nodiscard]] static SccResult all_compute_components(const DistributedSccState& state) {
        return SccCommon::all_assemble_result(SccCommon::all_compute_gathered_component_sizes(state));
    }
};
