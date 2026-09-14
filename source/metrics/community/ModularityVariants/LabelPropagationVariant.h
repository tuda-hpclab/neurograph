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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

/**
 * @brief Community detection after Raghavan, Albert, and Kumara (2007), "Near linear time algorithm
 *		to detect community structures in large-scale networks": every node starts with its own
 *		label and then adopts the label that carries the most arc weight in its neighborhood, over
 *		and over, until no node changes its label anymore. One sweep costs one pass over all arcs
 *		and there are neither aggregation levels nor a modularity computation, which makes this by
 *		far the cheapest of the variants. It also optimizes nothing directly, though: the labels
 *		only follow the local majority, so the modularity of the partition is usually lower than
 *		the one Louvain and Leiden reach, and on dense graphs one label frequently takes over the
 *		whole graph, which leaves a single community of modularity zero.
 *		This deterministic variant sweeps the nodes in ascending id order, compares the labels in
 *		ascending order, and adopts a label only if it carries strictly more weight than the
 *		current one (the classic algorithm uses a random order and breaks ties randomly). Because
 *		every adopted label strictly increases the arc weight that runs inside the labels, and that
 *		weight is bounded by the total weight of the graph, the sweeps always terminate.
 */
class LabelPropagationVariant {
public:
    /**
     * @brief Detects the communities of the graph.
     * @param graph The graph
     * @return The community id per node, compacted to 0, ..., k - 1
     */
    [[nodiscard]] static std::vector<std::size_t> detect_communities(const CommunityGraph& graph) {
        const auto number_nodes = graph.get_number_nodes();

        auto labels = std::vector<std::size_t>(number_nodes);
        std::iota(labels.begin(), labels.end(), std::size_t{ 0 });

        auto label_weights = std::vector<double>(number_nodes, 0.0);
        auto candidates = std::vector<std::size_t>{};

        auto changed_any_label = true;

        while (changed_any_label) {
            changed_any_label = false;

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                // A self loop carries its weight into whichever label the node currently has and
                // would do so for every label it could adopt, so it cannot change the majority.
                for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                    if (target == node_id) {
                        continue;
                    }

                    const auto label = labels[target];
                    if (label_weights[label] == 0.0) {
                        candidates.push_back(label);
                    }
                    label_weights[label] += weight;
                }

                for (const auto& [source, weight] : graph.get_in_arcs(node_id)) {
                    if (source == node_id) {
                        continue;
                    }

                    const auto label = labels[source];
                    if (label_weights[label] == 0.0) {
                        candidates.push_back(label);
                    }
                    label_weights[label] += weight;
                }

                std::ranges::sort(candidates);

                auto best_label = labels[node_id];
                auto best_weight = label_weights[best_label];

                for (const auto candidate : candidates) {
                    if (label_weights[candidate] > best_weight) {
                        best_weight = label_weights[candidate];
                        best_label = candidate;
                    }
                }

                if (best_label != labels[node_id]) {
                    labels[node_id] = best_label;
                    changed_any_label = true;
                }

                for (const auto candidate : candidates) {
                    label_weights[candidate] = 0.0;
                }
                candidates.clear();
            }
        }

        // The labels are node ids, so they have to be renumbered into community ids
        static_cast<void>(CommunityDetectionCommon::compact_communities(labels));

        return labels;
    }
};
