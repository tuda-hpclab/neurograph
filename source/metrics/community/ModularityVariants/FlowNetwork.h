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
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace infomap {

/**
 * @brief The random walk that InfomapVariant compresses: how much time the walker spends on each
 *		node (the node flows) and how often it uses each arc (the link flows). Both are probabilities;
 *		the node flows sum to one, and so do the link flows.
 *		The node flows are the stationary distribution of a walker that follows the arc weights and
 *		teleports to a uniformly drawn node with a fixed probability, i.e., a PageRank. The
 *		teleportation is what makes the flows well defined on a directed graph at all, because
 *		without it all flow would pile up in the sinks. It does not appear in the link flows,
 *		however: only the steps along an arc are recorded, and the link flows are renormalized to
 *		sum to one afterwards, so that a module's exit flow stays a property of the arcs and does
 *		not grow with the number of nodes outside the module.
 *		Self loops are dropped, because a step onto the node itself neither moves the walker nor
 *		crosses a module boundary; a network therefore never contains a link from a node to itself.
 */
class FlowNetwork {
public:
    /** A link as (node id, flow). */
    using Link = std::pair<std::size_t, double>;

    /**
     * @brief Computes the flows of the random walk on the given graph.
     * @param graph The graph; all arc weights must be positive
     * @param teleportation_probability The probability that a step teleports instead of following
     *		an arc, in (0, 1)
     * @return The flow network, with one node per node of the graph
     */
    [[nodiscard]] static FlowNetwork from_community_graph(const CommunityGraph& graph, const double teleportation_probability) {
        const auto number_nodes = graph.get_number_nodes();

        auto network = FlowNetwork{ number_nodes };

        if (number_nodes == 0) {
            return network;
        }

        // The strengths exclude the self loops, so that the walker never steps onto its own node
        auto out_strengths = std::vector<double>(number_nodes, 0.0);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                if (target != node_id) {
                    out_strengths[node_id] += weight;
                }
            }
        }

        network.node_flows = compute_node_flows(graph, out_strengths, teleportation_probability);

        // Every node with at least one arc spreads its whole flow over its arcs, so the flow on all
        // arcs together is the flow of exactly those nodes. Summing it up front avoids that the
        // renormalization below depends on the order in which the arcs are visited.
        auto total_link_flow = 0.0;

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            if (out_strengths[node_id] > 0.0) {
                total_link_flow += network.node_flows[node_id];
            }
        }

        if (total_link_flow == 0.0) {
            // A graph whose nodes are all isolated; the walker only ever teleports
            return network;
        }

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            if (out_strengths[node_id] == 0.0) {
                continue;
            }

            const auto flow_per_weight = network.node_flows[node_id] / (out_strengths[node_id] * total_link_flow);

            for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                if (target != node_id) {
                    network.add_link(node_id, target, flow_per_weight * weight);
                }
            }
        }

        return network;
    }

    /**
     * @brief Collapses every module into one node, summing up the flows. The links inside a module
     *		are dropped: they cannot leave any module of the aggregated network either, and only the
     *		flow that leaves a module enters the map equation.
     * @param assignment The module id per node, compacted to 0, ..., number_modules - 1
     * @param number_modules The number of distinct modules
     * @return The aggregated network with one node per module
     */
    [[nodiscard]] FlowNetwork aggregate(const std::span<const std::size_t> assignment, const std::size_t number_modules) const {
        const auto number_nodes = get_number_nodes();

        auto aggregated = FlowNetwork{ number_modules };

        auto module_members = std::vector<std::vector<std::size_t>>(number_modules);

        for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
            aggregated.node_flows[assignment[node_id]] += node_flows[node_id];
            module_members[assignment[node_id]].push_back(node_id);
        }

        auto flow_to = std::vector<double>(number_modules, 0.0);
        auto touched = std::vector<std::size_t>{};

        for (auto module_id = std::size_t{ 0 }; module_id < number_modules; ++module_id) {
            for (const auto member : module_members[module_id]) {
                for (const auto& [target, flow] : get_out_links(member)) {
                    const auto target_module = assignment[target];
                    if (target_module == module_id) {
                        continue;
                    }

                    if (flow_to[target_module] == 0.0) {
                        touched.push_back(target_module);
                    }
                    flow_to[target_module] += flow;
                }
            }

            std::ranges::sort(touched);

            for (const auto target_module : touched) {
                aggregated.add_link(module_id, target_module, flow_to[target_module]);
                flow_to[target_module] = 0.0;
            }
            touched.clear();
        }

        return aggregated;
    }

    /**
     * @brief Returns the number of nodes
     * @return The number of nodes
     */
    [[nodiscard]] std::size_t get_number_nodes() const noexcept {
        return node_flows.size();
    }

    /**
     * @brief Returns the flow of the specified node, i.e., how much time the walker spends on it
     * @param node_id The node id
     * @return The node's flow
     */
    [[nodiscard]] double get_flow(const std::size_t node_id) const noexcept {
        return node_flows[node_id];
    }

    /**
     * @brief Returns the flow that leaves the specified node, i.e., the total flow of its out links.
     *		This is not the node's flow, because the flow of the teleportation steps is not on any link.
     * @param node_id The node id
     * @return The flow on the node's out links
     */
    [[nodiscard]] double get_out_flow(const std::size_t node_id) const noexcept {
        return out_flows[node_id];
    }

    /**
     * @brief Returns the out links of the specified node
     * @param node_id The node id
     * @return The out links as (target, flow) pairs
     */
    [[nodiscard]] std::span<const Link> get_out_links(const std::size_t node_id) const noexcept {
        return node_out_links[node_id];
    }

    /**
     * @brief Returns the in links of the specified node
     * @param node_id The node id
     * @return The in links as (source, flow) pairs
     */
    [[nodiscard]] std::span<const Link> get_in_links(const std::size_t node_id) const noexcept {
        return node_in_links[node_id];
    }

private:
    /** The power iteration stops once no node flow changes by more than this much anymore. */
    static constexpr auto convergence_tolerance = 1e-13;

    /** An upper bound on the power iterations, so that a graph that converges slowly still terminates. */
    static constexpr auto maximum_number_iterations = std::size_t{ 1000 };

    explicit FlowNetwork(const std::size_t number_nodes)
        : node_flows(number_nodes, 0.0), node_out_links(number_nodes), node_in_links(number_nodes), out_flows(number_nodes, 0.0) {
    }

    void add_link(const std::size_t source, const std::size_t target, const double flow) {
        node_out_links[source].emplace_back(target, flow);
        node_in_links[target].emplace_back(source, flow);

        out_flows[source] += flow;
    }

    /**
     * @brief Computes the stationary distribution of the walker with a power iteration. A node
     *		without out arcs, i.e., a sink, teleports in every step, which keeps the distribution a
     *		probability distribution.
     * @param graph The graph
     * @param out_strengths The out strength per node, excluding the self loops
     * @param teleportation_probability The probability that a step teleports instead of following an arc
     * @return The flow per node; sums to one
     */
    [[nodiscard]] static std::vector<double> compute_node_flows(const CommunityGraph& graph, const std::span<const double> out_strengths,
                                                                const double teleportation_probability) {
        const auto number_nodes = graph.get_number_nodes();
        const auto uniform_flow = 1.0 / static_cast<double>(number_nodes);

        auto flows = std::vector<double>(number_nodes, uniform_flow);
        auto next_flows = std::vector<double>(number_nodes, 0.0);

        for (auto iteration = std::size_t{ 0 }; iteration < maximum_number_iterations; ++iteration) {
            auto sink_flow = 0.0;

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                if (out_strengths[node_id] == 0.0) {
                    sink_flow += flows[node_id];
                }
            }

            const auto teleported_flow = (teleportation_probability + (1.0 - teleportation_probability) * sink_flow) * uniform_flow;
            std::ranges::fill(next_flows, teleported_flow);

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                if (out_strengths[node_id] == 0.0) {
                    continue;
                }

                const auto flow_per_weight = (1.0 - teleportation_probability) * flows[node_id] / out_strengths[node_id];

                for (const auto& [target, weight] : graph.get_out_arcs(node_id)) {
                    if (target != node_id) {
                        next_flows[target] += flow_per_weight * weight;
                    }
                }
            }

            auto largest_change = 0.0;

            for (auto node_id = std::size_t{ 0 }; node_id < number_nodes; ++node_id) {
                largest_change = std::max(largest_change, std::abs(next_flows[node_id] - flows[node_id]));
            }

            flows.swap(next_flows);

            if (largest_change < convergence_tolerance) {
                break;
            }
        }

        return flows;
    }

    std::vector<double> node_flows;

    std::vector<std::vector<Link>> node_out_links;
    std::vector<std::vector<Link>> node_in_links;

    std::vector<double> out_flows;
};

} // namespace infomap
