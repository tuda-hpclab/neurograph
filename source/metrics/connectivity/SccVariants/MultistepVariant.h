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

#include <unordered_map>

/**
 * @brief Strongly connected components via the multistep method by Slota, Rajamanickam, and Madduri
 *		(2014), "BFS and coloring-based parallel algorithms for strongly connected components and
 *		related problems", on the distributed graph. It combines the phases of the other variants in
 *		the order in which each of them is cheapest:
 *
 *		(1) Trimming removes the nodes that cannot lie on a cycle at all, which are the vast majority
 *			in a graph whose components are mostly single nodes, at the cost of one exchange per layer.
 *		(2) One forward-backward round peels the component of the node with the largest degree product,
 *			which on a small-world graph such as a connectome is the one large component that holds
 *			most of the remaining nodes.
 *		(3) The coloring rounds finish the remaining small components, where one round handles a whole
 *			color layer instead of a single component.
 *		(4) Once few enough nodes are left, the rest is gathered on the root rank and finished with the
 *			sequential algorithm, which spends one collective exchange instead of a round per layer.
 *			The threshold is chosen by the caller because the gathered part has to fit into the memory
 *			of one rank; a threshold of zero skips this phase and keeps the computation distributed.
 *
 *		Since removing a component leaves the other components untouched, every phase may hand its
 *		leftovers to the next one, and the sizes the phases found simply add up.
 */
class MultistepVariant {
public:
    /**
     * @brief Determines the strongly connected components of the whole graph.
     *		Must be called on every MPI rank with the same threshold, returns the same result on every
     *		MPI rank.
     * @param state The state, which is fully processed afterwards unless the last phase gathered the rest
     * @param tail_threshold How many nodes may be left over for the root rank to finish sequentially;
     *		zero keeps every phase distributed
     * @return The components, summarized by their sizes
     */
    [[nodiscard]] static SccResult all_compute_components(DistributedSccState& state, const global_node_id_type tail_threshold) {
        state.all_trim();

        const auto pivot = state.all_select_pivot();
        if (!pivot.has_uninitialized_rank()) {
            state.all_assign_component_of(pivot);
        }

        // The sizes the root rank finds for the gathered rest; the other phases store their components
        // in the state itself
        auto size_histogram = std::unordered_map<global_node_id_type, global_node_id_type>{};

        while (true) {
            const auto number_active_nodes = state.all_count_active_nodes();

            if (number_active_nodes == 0) {
                break;
            }

            if (number_active_nodes <= tail_threshold) {
                size_histogram = SccCommon::all_compute_gathered_component_sizes(state);
                break;
            }

            state.all_assign_components_by_coloring();
        }

        for (const auto& [number_nodes_in_component, number_components] : state.all_count_component_sizes()) {
            size_histogram[number_nodes_in_component] += number_components;
        }

        return SccCommon::all_assemble_result(size_histogram);
    }
};
