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

#include "metrics/community/ModularityVariants/CommunityGraph.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

/**
 * @brief Shared building blocks of the modularity community-detection variants (see the sibling
 *		headers in metrics/community/ModularityVariants). Everything works on the directed
 *		modularity by Leicht and Newman (2008),
 *			Q = sum over communities c of [ w_cc / m - (sigma_out_c * sigma_in_c) / m^2 ],
 *		where w_cc is the weight inside c, sigma_out/in_c are the summed node strengths of c, and
 *		m is the total arc weight. On symmetric graphs this equals the classic undirected modularity.
 *		All steps are deterministic: nodes are visited in ascending id order, candidate communities
 *		are compared in ascending id order, and only a strictly larger gain wins. The gain compares
 *		integer-valued sums (see CommunityGraph), so the decisions do not depend on the platform,
 *		the compiler, or the number of MPI ranks.
 */
class CommunityDetectionCommon {
public:
    /**
     * @brief Improves the given community assignment by repeatedly sweeping over all nodes and
     *		moving each node to the neighboring community with the highest modularity gain,
     *		until a full sweep moves no node (the local-moving phase of Louvain and Leiden).
     *		A node moves only if the gain is strictly larger than the gain of staying, so ties
     *		keep the current community and the sweep terminates.
     * @param graph The graph
     * @param assignment The community id per node; both read and updated. Ids must be smaller
     *		than the number of nodes.
     * @return True iff at least one node changed its community
     */
    [[nodiscard]] static bool local_move(const CommunityGraph& graph, std::vector<std::size_t>& assignment) {
        const auto number_nodes = graph.get_number_nodes();
        const auto total_weight = graph.get_total_weight();

        auto sigma_out = std::vector<double>(number_nodes, 0.0);
        auto sigma_in = std::vector<double>(number_nodes, 0.0);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            sigma_out[assignment[node_id]] += graph.get_out_strength(node_id);
            sigma_in[assignment[node_id]] += graph.get_in_strength(node_id);
        }

        auto weight_to = std::vector<double>(number_nodes, 0.0);
        auto weight_from = std::vector<double>(number_nodes, 0.0);
        auto candidates = std::vector<std::size_t>{};

        auto moved_any_node = false;
        auto moved = true;

        while (moved) {
            moved = false;

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                const auto old_community = assignment[node_id];

                // Self loops move together with the node and gain the same amount in every
                // community, so they are excluded from the neighbor-community weights.
                for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                    if (target == node_id) {
                        continue;
                    }

                    const auto community = assignment[target];
                    if (weight_to[community] == 0.0 && weight_from[community] == 0.0) {
                        candidates.push_back(community);
                    }
                    weight_to[community] += weight;
                }

                for (const auto& [source, weight] : graph.get_in_arcs(node_id)) {
                    if (source == node_id) {
                        continue;
                    }

                    const auto community = assignment[source];
                    if (weight_to[community] == 0.0 && weight_from[community] == 0.0) {
                        candidates.push_back(community);
                    }
                    weight_from[community] += weight;
                }

                // The gain formulas compare the node against a community it is not part of,
                // so remove the node from its own community first.
                sigma_out[old_community] -= graph.get_out_strength(node_id);
                sigma_in[old_community] -= graph.get_in_strength(node_id);

                const auto gain_of_joining = [&](const std::size_t community) {
                    return (weight_to[community] + weight_from[community])
                           - (graph.get_out_strength(node_id) * sigma_in[community] + graph.get_in_strength(node_id) * sigma_out[community]) / total_weight;
                };

                std::ranges::sort(candidates);

                auto best_community = old_community;
                auto best_gain = gain_of_joining(old_community);

                for (const auto candidate : candidates) {
                    const auto gain = gain_of_joining(candidate);
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_community = candidate;
                    }
                }

                sigma_out[best_community] += graph.get_out_strength(node_id);
                sigma_in[best_community] += graph.get_in_strength(node_id);
                assignment[node_id] = best_community;

                if (best_community != old_community) {
                    moved = true;
                    moved_any_node = true;
                }

                for (const auto candidate : candidates) {
                    weight_to[candidate] = 0.0;
                    weight_from[candidate] = 0.0;
                }
                candidates.clear();
            }
        }

        return moved_any_node;
    }

    /**
     * @brief Refines a community assignment into a finer partition (the refinement phase of
     *		Leiden). Every node starts in its own refined community; a node that is still alone
     *		merges into the refined community inside its own community with the highest strictly
     *		positive modularity gain. This is the deterministic greedy special case of the Leiden
     *		refinement (randomness parameter towards zero).
     * @param graph The graph
     * @param assignment The community id per node, as produced by local_move
     * @return The refined community id per node; refines assignment, i.e., every refined
     *		community lies completely inside one community
     */
    [[nodiscard]] static std::vector<std::size_t> refine(const CommunityGraph& graph, const std::span<const std::size_t> assignment) {
        const auto number_nodes = graph.get_number_nodes();
        const auto total_weight = graph.get_total_weight();

        auto refined = std::vector<std::size_t>(number_nodes);
        auto community_sizes = std::vector<std::size_t>(number_nodes, 1);
        auto sigma_out = std::vector<double>(number_nodes, 0.0);
        auto sigma_in = std::vector<double>(number_nodes, 0.0);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            refined[node_id] = node_id;
            sigma_out[node_id] = graph.get_out_strength(node_id);
            sigma_in[node_id] = graph.get_in_strength(node_id);
        }

        auto weight_to = std::vector<double>(number_nodes, 0.0);
        auto weight_from = std::vector<double>(number_nodes, 0.0);
        auto candidates = std::vector<std::size_t>{};

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            if (community_sizes[refined[node_id]] > 1) {
                // The node already merged with an earlier node; only nodes that are still alone
                // may move, which keeps every merge decision final and the result deterministic.
                continue;
            }

            for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                if (target == node_id || assignment[target] != assignment[node_id]) {
                    continue;
                }

                const auto refined_community = refined[target];
                if (weight_to[refined_community] == 0.0 && weight_from[refined_community] == 0.0) {
                    candidates.push_back(refined_community);
                }
                weight_to[refined_community] += weight;
            }

            for (const auto& [source, weight] : graph.get_in_arcs(node_id)) {
                if (source == node_id || assignment[source] != assignment[node_id]) {
                    continue;
                }

                const auto refined_community = refined[source];
                if (weight_to[refined_community] == 0.0 && weight_from[refined_community] == 0.0) {
                    candidates.push_back(refined_community);
                }
                weight_from[refined_community] += weight;
            }

            std::ranges::sort(candidates);

            auto best_community = refined[node_id];
            auto best_gain = 0.0;

            for (const auto candidate : candidates) {
                if (candidate == refined[node_id]) {
                    continue;
                }

                const auto gain = (weight_to[candidate] + weight_from[candidate])
                                  - (graph.get_out_strength(node_id) * sigma_in[candidate] + graph.get_in_strength(node_id) * sigma_out[candidate]) / total_weight;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_community = candidate;
                }
            }

            if (best_community != refined[node_id]) {
                sigma_out[refined[node_id]] -= graph.get_out_strength(node_id);
                sigma_in[refined[node_id]] -= graph.get_in_strength(node_id);
                sigma_out[best_community] += graph.get_out_strength(node_id);
                sigma_in[best_community] += graph.get_in_strength(node_id);

                --community_sizes[refined[node_id]];
                ++community_sizes[best_community];
                refined[node_id] = best_community;
            }

            for (const auto candidate : candidates) {
                weight_to[candidate] = 0.0;
                weight_from[candidate] = 0.0;
            }
            candidates.clear();
        }

        return refined;
    }

    /**
     * @brief Renumbers the community ids to 0, ..., k - 1, keeping their ascending order.
     * @param assignment The community id per node; updated in place. Ids must be smaller
     *		than the number of nodes.
     * @return The number of distinct communities k
     */
    static std::size_t compact_communities(std::vector<std::size_t>& assignment) {
        auto id_is_used = std::vector<char>(assignment.size(), 0);
        for (const auto community : assignment) {
            id_is_used[community] = 1;
        }

        auto new_ids = std::vector<std::size_t>(assignment.size(), 0);
        auto number_communities = std::size_t{ 0 };

        for (auto id = std::size_t{ 0 }; id < id_is_used.size(); ++id) {
            if (id_is_used[id] == 1) {
                new_ids[id] = number_communities;
                ++number_communities;
            }
        }

        for (auto& community : assignment) {
            community = new_ids[community];
        }

        return number_communities;
    }

    /**
     * @brief Collapses every community into one node. Arc weights between two communities are
     *		summed into one arc; the weight inside a community becomes a self loop, so the
     *		aggregated graph keeps the same total weight and the same modularity landscape.
     * @param graph The graph
     * @param assignment The community id per node, compacted to 0, ..., number_communities - 1
     * @param number_communities The number of distinct communities
     * @return The aggregated graph with one node per community
     */
    [[nodiscard]] static CommunityGraph aggregate(const CommunityGraph& graph, const std::span<const std::size_t> assignment, const std::size_t number_communities) {
        const auto number_nodes = graph.get_number_nodes();

        auto community_members = std::vector<std::vector<std::size_t>>(number_communities);
        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            community_members[assignment[node_id]].push_back(node_id);
        }

        auto aggregated = CommunityGraph{ number_communities };

        auto weight_to = std::vector<double>(number_communities, 0.0);
        auto touched = std::vector<std::size_t>{};

        for (auto community = std::size_t{ 0 }; community < number_communities; ++community) {
            for (const auto member : community_members[community]) {
                for (const auto& [target, weight] : graph.get_out_arcs(member)) {
                    const auto target_community = assignment[target];
                    if (weight_to[target_community] == 0.0) {
                        touched.push_back(target_community);
                    }
                    weight_to[target_community] += weight;
                }
            }

            std::ranges::sort(touched);

            for (const auto target_community : touched) {
                aggregated.add_arc(community, target_community, weight_to[target_community]);
                weight_to[target_community] = 0.0;
            }
            touched.clear();
        }

        return aggregated;
    }

    /**
     * @brief Computes the directed modularity of the given community assignment.
     * @param graph The graph
     * @param assignment The community id per node. Ids must be smaller than the number of nodes.
     * @return The modularity, or 0.0 for a graph without arcs
     */
    [[nodiscard]] static double compute_modularity(const CommunityGraph& graph, const std::span<const std::size_t> assignment) {
        const auto number_nodes = graph.get_number_nodes();
        const auto total_weight = graph.get_total_weight();

        if (total_weight == 0.0) {
            return 0.0;
        }

        auto intra_community_weights = std::vector<double>(number_nodes, 0.0);
        auto sigma_out = std::vector<double>(number_nodes, 0.0);
        auto sigma_in = std::vector<double>(number_nodes, 0.0);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                if (assignment[node_id] == assignment[target]) {
                    intra_community_weights[assignment[node_id]] += weight;
                }
            }

            sigma_out[assignment[node_id]] += graph.get_out_strength(node_id);
            sigma_in[assignment[node_id]] += graph.get_in_strength(node_id);
        }

        auto modularity = 0.0;

        // Empty community ids contribute zero to both terms, so all ids can be summed uniformly.
        for (auto community = std::size_t{ 0 }; community < number_nodes; ++community) {
            modularity += intra_community_weights[community] / total_weight - (sigma_out[community] * sigma_in[community]) / (total_weight * total_weight);
        }

        return modularity;
    }
};
