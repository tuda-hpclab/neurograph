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
#include <cstdint>
#include <span>
#include <vector>

namespace max_flow {

/**
 * @brief A residual flow network over contiguous node ids 0, ..., n - 1, used by the maximum-flow
 *		variants (see the types in metrics/flow/MaxFlowVariants). Every arc is stored together with
 *		a reverse arc of capacity zero; pushing flow over an arc moves that amount of capacity onto
 *		its reverse arc, so the stored capacities always describe the current residual network and
 *		pushed flow stays revocable.
 *		All capacities are non-negative integers, so every augmentation is integral and the
 *		variants terminate with the exact maximum flow.
 */
class FlowNetwork {
public:
    /** The capacity type; large enough to hold the sum of all arc capacities. */
    using capacity_type = std::int64_t;

    /** One direction of an arc in the residual network. */
    struct ResidualArc {
        std::size_t target{};
        capacity_type capacity{};
        std::size_t reverse_arc_id{};
    };

    /**
     * @brief Constructs a network with the given number of nodes and no arcs.
     * @param number_nodes The number of nodes
     */
    explicit FlowNetwork(const std::size_t number_nodes)
        : node_arc_ids(number_nodes) {
    }

    /**
     * @brief Adds one directed arc and its reverse arc with capacity zero. Parallel arcs are
     *		allowed and act like one arc with the summed capacity.
     * @param source The source node id
     * @param target The target node id
     * @param capacity The non-negative arc capacity
     */
    void add_arc(const std::size_t source, const std::size_t target, const capacity_type capacity) {
        const auto arc_id = arcs.size();

        node_arc_ids[source].push_back(arc_id);
        arcs.push_back(ResidualArc{ target, capacity, arc_id + 1 });

        node_arc_ids[target].push_back(arc_id + 1);
        arcs.push_back(ResidualArc{ source, 0, arc_id });
    }

    /**
     * @brief Returns the number of nodes
     * @return The number of nodes
     */
    [[nodiscard]] std::size_t get_number_nodes() const noexcept {
        return node_arc_ids.size();
    }

    /**
     * @brief Returns the ids of all arcs that leave the node in the residual network,
     *		which includes the reverse arcs of the arcs that enter it
     * @param node_id The node id
     * @return The ids of the node's residual arcs, in insertion order
     */
    [[nodiscard]] std::span<const std::size_t> get_arc_ids(const std::size_t node_id) const noexcept {
        return node_arc_ids[node_id];
    }

    /**
     * @brief Returns the specified residual arc
     * @param arc_id The arc id
     * @return The arc with its target, its remaining capacity, and the id of its reverse arc
     */
    [[nodiscard]] const ResidualArc& get_arc(const std::size_t arc_id) const noexcept {
        return arcs[arc_id];
    }

    /**
     * @brief Pushes flow over the arc: reduces its residual capacity and grows the capacity of
     *		its reverse arc by the same amount
     * @param arc_id The arc id
     * @param amount The flow; must not exceed the arc's residual capacity
     */
    void push_flow(const std::size_t arc_id, const capacity_type amount) {
        arcs[arc_id].capacity -= amount;
        arcs[arcs[arc_id].reverse_arc_id].capacity += amount;
    }

private:
    std::vector<ResidualArc> arcs;
    std::vector<std::vector<std::size_t>> node_arc_ids;
};

} // namespace max_flow
