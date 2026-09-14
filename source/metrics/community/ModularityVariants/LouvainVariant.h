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
#include <vector>

/**
 * @brief Community detection after Blondel et al. (2008), "Fast unfolding of communities in large
 *		networks": starting from singleton communities, greedily move nodes to the neighboring
 *		community with the highest modularity gain until no move improves the modularity, then
 *		collapse every community into one node and repeat on the aggregated graph.
 *		This deterministic variant sweeps the nodes in ascending id order (the classic algorithm
 *		visits them in random order), so it always returns the same partition for the same graph.
 *		Every accepted move strictly increases the modularity, but the result is only a local
 *		maximum; communities can in rare cases end up internally disconnected, which the Leiden
 *		variant fixes.
 */
class LouvainVariant {
public:
    /**
     * @brief Detects the communities of the graph.
     * @param graph The graph
     * @return The community id per node, compacted to 0, ..., k - 1
     */
    [[nodiscard]] static std::vector<std::size_t> detect_communities(const CommunityGraph& graph) {
        auto node_to_community = std::vector<std::size_t>(graph.get_number_nodes());
        std::iota(node_to_community.begin(), node_to_community.end(), std::size_t{ 0 });

        // The first level runs on the input graph, every further level on the previous aggregation.
        auto aggregated = std::optional<CommunityGraph>{};

        while (true) {
            const auto& level_graph = aggregated.has_value() ? *aggregated : graph;

            auto assignment = std::vector<std::size_t>(level_graph.get_number_nodes());
            std::iota(assignment.begin(), assignment.end(), std::size_t{ 0 });

            const auto moved_any_node = CommunityDetectionCommon::local_move(level_graph, assignment);
            if (!moved_any_node) {
                // The singleton partition of the aggregated graph is locally optimal; further
                // levels would aggregate the identical graph again, so the algorithm is done.
                break;
            }

            const auto number_communities = CommunityDetectionCommon::compact_communities(assignment);

            for (auto& community : node_to_community) {
                community = assignment[community];
            }

            aggregated = CommunityDetectionCommon::aggregate(level_graph, assignment, number_communities);
        }

        return node_to_community;
    }
};
