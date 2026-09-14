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

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief A weighted directed graph over contiguous node ids 0, ..., n - 1, used by the modularity
 *		community-detection variants (see the types in metrics/community/ModularityVariants).
 *		Stores both adjacency directions plus the per-node strengths (weighted degrees), because
 *		the local-moving heuristic scans the in and the out neighbors of a node and weighs its
 *		strengths against the community strengths.
 *		All arc weights must be positive. In practice they are (sums of) integer graph weights,
 *		so every strength and community weight is an exact integer in a double and the community
 *		detection decides identically across platforms and compilers.
 */
class CommunityGraph {
public:
    /** A neighbor as (node id, arc weight). */
    using Neighbor = std::pair<std::size_t, double>;

    /**
     * @brief Constructs a graph with the given number of nodes and no arcs.
     * @param number_nodes The number of nodes
     */
    explicit CommunityGraph(const std::size_t number_nodes)
        : node_out_arcs(number_nodes), node_in_arcs(number_nodes), out_strengths(number_nodes, 0.0), in_strengths(number_nodes, 0.0) {
    }

    /**
     * @brief Adds one directed arc. Parallel arcs are allowed and act like one arc with the summed weight;
     *		a self loop counts once towards the total weight and once towards each of its node's strengths.
     * @param source The source node id
     * @param target The target node id
     * @param weight The positive arc weight
     */
    void add_arc(const std::size_t source, const std::size_t target, const double weight) {
        node_out_arcs[source].emplace_back(target, weight);
        node_in_arcs[target].emplace_back(source, weight);

        out_strengths[source] += weight;
        in_strengths[target] += weight;
        total_weight += weight;
    }

    /**
     * @brief Returns the number of nodes
     * @return The number of nodes
     */
    [[nodiscard]] std::size_t get_number_nodes() const noexcept {
        return node_out_arcs.size();
    }

    /**
     * @brief Returns the out arcs of the specified node
     * @param node_id The node id
     * @return The out arcs as (target, weight) pairs
     */
    [[nodiscard]] std::span<const Neighbor> get_out_arcs(const std::size_t node_id) const noexcept {
        return node_out_arcs[node_id];
    }

    /**
     * @brief Returns the in arcs of the specified node
     * @param node_id The node id
     * @return The in arcs as (source, weight) pairs
     */
    [[nodiscard]] std::span<const Neighbor> get_in_arcs(const std::size_t node_id) const noexcept {
        return node_in_arcs[node_id];
    }

    /**
     * @brief Returns the out strength of the specified node, i.e., the total weight of its out arcs
     * @param node_id The node id
     * @return The node's out strength
     */
    [[nodiscard]] double get_out_strength(const std::size_t node_id) const noexcept {
        return out_strengths[node_id];
    }

    /**
     * @brief Returns the in strength of the specified node, i.e., the total weight of its in arcs
     * @param node_id The node id
     * @return The node's in strength
     */
    [[nodiscard]] double get_in_strength(const std::size_t node_id) const noexcept {
        return in_strengths[node_id];
    }

    /**
     * @brief Returns the total weight of all arcs
     * @return The total weight
     */
    [[nodiscard]] double get_total_weight() const noexcept {
        return total_weight;
    }

private:
    std::vector<std::vector<Neighbor>> node_out_arcs;
    std::vector<std::vector<Neighbor>> node_in_arcs;

    std::vector<double> out_strengths;
    std::vector<double> in_strengths;

    double total_weight{ 0.0 };
};
