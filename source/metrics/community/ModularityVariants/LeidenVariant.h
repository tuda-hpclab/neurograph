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

#include "metrics/community/ModularityVariants/CommunityDetectionCommon.h"
#include "metrics/community/ModularityVariants/CommunityGraph.h"

#include <cstddef>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

/**
 * @brief Community detection after Traag, Waltman, and van Eck (2019), "From Louvain to Leiden:
 *		guaranteeing well-connected communities": like Louvain, but between the local moving and
 *		the aggregation the communities are refined into finer, internally connected parts. The
 *		graph is aggregated over the refined partition while the next level starts from the
 *		coarser communities, so badly connected communities are split up again instead of being
 *		frozen into single nodes.
 *		This deterministic variant sweeps the nodes in ascending id order and merges refinement
 *		singletons greedily by the best gain (the paper picks randomly among the positive gains),
 *		so it always returns the same partition for the same graph.
 */
class LeidenVariant {
public:
    /**
     * @brief Detects the communities of the graph.
     * @param graph The graph
     * @return The community id per node, compacted to 0, ..., k - 1
     */
    [[nodiscard]] static std::vector<std::size_t> detect_communities(const CommunityGraph& graph) {
        auto node_to_community = std::vector<std::size_t>(graph.get_number_nodes());
        std::iota(node_to_community.begin(), node_to_community.end(), std::size_t{ 0 });

        auto initial_assignment = node_to_community;

        // The first level runs on the input graph, every further level on the previous aggregation.
        auto aggregated = std::optional<CommunityGraph>{};

        while (true) {
            const auto& level_graph = aggregated.has_value() ? *aggregated : graph;

            auto assignment = initial_assignment;
            static_cast<void>(CommunityDetectionCommon::local_move(level_graph, assignment));

            auto compacted_assignment = assignment;
            const auto number_communities = CommunityDetectionCommon::compact_communities(compacted_assignment);

            if (number_communities == level_graph.get_number_nodes()) {
                // Every aggregated node forms its own community, so no further level can merge
                // anything; node_to_community already maps onto exactly these communities.
                break;
            }

            auto refined = CommunityDetectionCommon::refine(level_graph, assignment);
            auto number_refined = CommunityDetectionCommon::compact_communities(refined);

            if (number_refined == level_graph.get_number_nodes()) {
                // The refinement kept every node alone; aggregating over it would reproduce the
                // current graph. Aggregate over the communities instead (a plain Louvain level),
                // which strictly shrinks the graph because at least one community has two nodes.
#if defined(__GNUC__) && !defined(__clang__)
                // GCC 13 misreports a "potential null pointer dereference" inlined from
                // std::vector::operator= here; both vectors are always non-null, sized data.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
                refined = compacted_assignment;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
                number_refined = number_communities;
            }

            for (auto& community : node_to_community) {
                community = refined[community];
            }

            // The next level starts from the current communities: every refined part becomes one
            // node and is initially assigned to the community it was refined out of.
            auto next_initial_assignment = std::vector<std::size_t>(number_refined);
            for (auto node_id = std::size_t{ 0 }; node_id < level_graph.get_number_nodes(); ++node_id) {
                next_initial_assignment[refined[node_id]] = compacted_assignment[node_id];
            }
            initial_assignment = std::move(next_initial_assignment);

            aggregated = CommunityDetectionCommon::aggregate(level_graph, refined, number_refined);
        }

        return node_to_community;
    }
};
