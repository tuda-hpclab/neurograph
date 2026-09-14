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
#include "metrics/community/ModularityVariants/FlowNetwork.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

/**
 * @brief Community detection after Rosvall and Bergstrom (2008), "Maps of random walks on complex
 *		networks reveal community structure": the communities are the modules that describe a random
 *		walk on the graph with as few bits as possible. The idea is that a walker stays inside a
 *		community for a long time, so naming the community only when the walker leaves it and using
 *		a separate, short code book inside every community compresses the walk. The description
 *		length of that two-level code is the map equation
 *			L = plogp(q) - 2 * sum over modules i of plogp(q_i) - sum over nodes a of plogp(p_a)
 *				+ sum over modules i of plogp(q_i + p_i),
 *		with plogp(x) = x * log2(x), the flow p_a of a node, the flow p_i and the exit flow q_i of a
 *		module, and the total exit flow q. The flows come from infomap::FlowNetwork.
 *		Unlike the other variants this optimizes the description length and not the modularity, so
 *		the reported modularity of its partition is usually lower: the partition answers a different
 *		question. What it sees instead of the arc weights around a node is where the flow of a walk
 *		along the arc directions goes, which does not depend on the density of the graph and is
 *		therefore not subject to the resolution limit of the modularity.
 *		The modules are searched greedily in levels like in Louvain: move single nodes as long as
 *		that shortens the description, then collapse every module into one node and repeat. This
 *		deterministic variant sweeps the nodes in ascending id order and compares the candidate
 *		modules in ascending id order; the original paper instead visits them in random order and
 *		refines the greedy result with simulated annealing.
 */
class InfomapVariant {
public:
    /**
     * @brief Detects the communities of the graph.
     * @param graph The graph
     * @return The community id per node, compacted to 0, ..., k - 1
     */
    [[nodiscard]] static std::vector<std::size_t> detect_communities(const CommunityGraph& graph) {
        auto node_to_module = std::vector<std::size_t>(graph.get_number_nodes());
        std::iota(node_to_module.begin(), node_to_module.end(), std::size_t{ 0 });

        const auto network = infomap::FlowNetwork::from_community_graph(graph, teleportation_probability);

        // The first level runs on the flow network itself, every further level on the previous aggregation.
        auto aggregated = std::optional<infomap::FlowNetwork>{};

        while (true) {
            const auto& level_network = aggregated.has_value() ? *aggregated : network;

            auto assignment = std::vector<std::size_t>(level_network.get_number_nodes());
            std::iota(assignment.begin(), assignment.end(), std::size_t{ 0 });

            const auto moved_any_node = local_move(level_network, assignment);
            if (!moved_any_node) {
                // Every node of the aggregated network is a module of its own and no single move
                // shortens the description, so no further level can compress the walk either.
                break;
            }

            const auto number_modules = CommunityDetectionCommon::compact_communities(assignment);

            for (auto& module_id : node_to_module) {
                module_id = assignment[module_id];
            }

            // The next level starts from the singletons of the aggregated network, so at least one
            // node has been merged away and the network strictly shrinks in every level.
            auto next_network = level_network.aggregate(assignment, number_modules);
            aggregated = std::move(next_network);
        }

        return node_to_module;
    }

private:
    /** The probability that a step of the walker teleports instead of following an arc. */
    static constexpr auto teleportation_probability = 0.15;

    /**
     * By how many bits a move has to shorten the description to be accepted. Unlike the modularity
     * gain of the other variants, the description length is not a sum of integers, so the moves are
     * only accepted with a margin against the rounding of the entropies.
     */
    static constexpr auto minimum_improvement = 1e-12;

    /**
     * An upper bound on the sweeps per level. Every accepted move shortens the description, which
     * bounds the sweeps in exact arithmetic already; the bound only guarantees that rounding cannot
     * keep the sweeps going forever.
     */
    static constexpr auto maximum_number_sweeps = std::size_t{ 100 };

    /**
     * @brief Computes x * log2(x), the contribution of a probability to an entropy.
     * @param value The probability x; a value of zero or, from rounding, slightly below contributes nothing
     * @return x * log2(x), or 0.0 for x <= 0
     */
    [[nodiscard]] static double plogp(const double value) {
        return value > 0.0 ? value * std::log2(value) : 0.0;
    }

    /**
     * @brief Computes what one module contributes to the map equation, i.e., everything except the
     *		term of the total exit flow and the term of the node flows, which no single module changes.
     * @param exit_flow The flow that leaves the module
     * @param flow The flow of the nodes inside the module
     * @return The contribution of the module; 0.0 for an empty module
     */
    [[nodiscard]] static double module_codelength(const double exit_flow, const double flow) {
        return plogp(exit_flow + flow) - 2.0 * plogp(exit_flow);
    }

    /**
     * @brief Improves the given module assignment by repeatedly sweeping over all nodes and moving
     *		each node into the neighboring module that shortens the description the most, until a
     *		full sweep moves no node. A node only moves if the description gets strictly shorter, so
     *		ties keep the current module.
     * @param network The flow network
     * @param assignment The module id per node; both read and updated. Ids must be smaller than the
     *		number of nodes.
     * @return True iff at least one node changed its module
     */
    [[nodiscard]] static bool local_move(const infomap::FlowNetwork& network, std::vector<std::size_t>& assignment) {
        const auto number_nodes = network.get_number_nodes();

        auto module_flows = std::vector<double>(number_nodes, 0.0);
        auto module_exit_flows = std::vector<double>(number_nodes, 0.0);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            module_flows[assignment[node_id]] += network.get_flow(node_id);

            for (const auto& [target, flow] : network.get_out_links(node_id)) {
                if (assignment[target] != assignment[node_id]) {
                    module_exit_flows[assignment[node_id]] += flow;
                }
            }
        }

        auto total_exit_flow = std::reduce(module_exit_flows.begin(), module_exit_flows.end(), 0.0);

        auto flow_to = std::vector<double>(number_nodes, 0.0);
        auto flow_from = std::vector<double>(number_nodes, 0.0);
        auto candidates = std::vector<std::size_t>{};

        auto moved_any_node = false;

        for (auto sweep = std::size_t{ 0 }; sweep < maximum_number_sweeps; ++sweep) {
            auto moved = false;

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                const auto old_module = assignment[node_id];

                // A flow network has no self links, so these are the links to and from other nodes
                for (const auto& [target, flow] : network.get_out_links(node_id)) {
                    const auto module_id = assignment[target];
                    if (flow_to[module_id] == 0.0 && flow_from[module_id] == 0.0) {
                        candidates.push_back(module_id);
                    }
                    flow_to[module_id] += flow;
                }

                for (const auto& [source, flow] : network.get_in_links(node_id)) {
                    const auto module_id = assignment[source];
                    if (flow_to[module_id] == 0.0 && flow_from[module_id] == 0.0) {
                        candidates.push_back(module_id);
                    }
                    flow_from[module_id] += flow;
                }

                // Once the node has left its module, its links out of the module do not leave that
                // module anymore, while the links between it and the rest of the module now do
                const auto exit_flow_without_node = module_exit_flows[old_module] - network.get_out_flow(node_id) + flow_to[old_module] + flow_from[old_module];
                const auto flow_without_node = module_flows[old_module] - network.get_flow(node_id);

                const auto codelength_of_old_module = module_codelength(module_exit_flows[old_module], module_flows[old_module]);
                const auto codelength_without_node = module_codelength(exit_flow_without_node, flow_without_node);
                const auto codelength_of_total_exit_flow = plogp(total_exit_flow);

                std::ranges::sort(candidates);

                auto best_module = old_module;
                auto best_difference = 0.0;
                auto best_exit_flow_with_node = 0.0;

                for (const auto candidate : candidates) {
                    if (candidate == old_module) {
                        continue;
                    }

                    // Mirrored: the links between the node and the module stop leaving it, the
                    // links out of the node into the rest of the graph start to
                    const auto exit_flow_with_node = module_exit_flows[candidate] + network.get_out_flow(node_id) - flow_to[candidate] - flow_from[candidate];
                    const auto flow_with_node = module_flows[candidate] + network.get_flow(node_id);

                    const auto next_total_exit_flow =
                        total_exit_flow - module_exit_flows[old_module] - module_exit_flows[candidate] + exit_flow_without_node + exit_flow_with_node;

                    const auto difference = plogp(next_total_exit_flow) - codelength_of_total_exit_flow + codelength_without_node
                                            + module_codelength(exit_flow_with_node, flow_with_node) - codelength_of_old_module
                                            - module_codelength(module_exit_flows[candidate], module_flows[candidate]);

                    if (difference < best_difference - minimum_improvement) {
                        best_difference = difference;
                        best_module = candidate;
                        best_exit_flow_with_node = exit_flow_with_node;
                    }
                }

                if (best_module != old_module) {
                    total_exit_flow += exit_flow_without_node + best_exit_flow_with_node - module_exit_flows[old_module] - module_exit_flows[best_module];

                    module_exit_flows[old_module] = exit_flow_without_node;
                    module_exit_flows[best_module] = best_exit_flow_with_node;
                    module_flows[old_module] = flow_without_node;
                    module_flows[best_module] += network.get_flow(node_id);

                    assignment[node_id] = best_module;

                    moved = true;
                    moved_any_node = true;
                }

                for (const auto candidate : candidates) {
                    flow_to[candidate] = 0.0;
                    flow_from[candidate] = 0.0;
                }
                candidates.clear();
            }

            if (!moved) {
                break;
            }
        }

        return moved_any_node;
    }
};
