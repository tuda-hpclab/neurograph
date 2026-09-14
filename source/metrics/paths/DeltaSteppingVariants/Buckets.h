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

#include <cpp-utility/Cast.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <unordered_set>
#include <vector>

/**
 * Implements the buckets used in delta stepping.
 * Holds a set of active nodes and list of buckets, each holding a set of nodes.
 */
class Buckets {
public:
    constexpr static std::size_t infinity_bucket_index = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Constructs the buckets with the given number of local nodes.
     * @param number_local_nodes The number of local nodes
     */
    explicit Buckets(const node_id_type number_local_nodes) {
        node_to_bucket.resize(utility::safe_cast<std::size_t>(number_local_nodes), infinity_bucket_index);
        infinity_bucket.reserve(utility::safe_cast<std::size_t>(number_local_nodes) * 2);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
            infinity_bucket.emplace(node_id);
        }
    }

    /**
     * @brief Returns the currently active nodes and clears them
     * @return The currently active nodes
     */
    [[nodiscard]] std::unordered_set<node_id_type> get_active_nodes() noexcept {
        auto temp_value = std::move(active_nodes);
        active_nodes.clear();
        return temp_value;
    }

    /**
     * @brief Moves a specified node to the new bucket. Does not mark the node as active.
     * @param node_id The local node
     * @param bucket_index The new bucket for the nodes
     */
    void move_node(const node_id_type node_id, const std::size_t bucket_index) {
        const auto previous_bucket_index = node_to_bucket[node_id];
        node_to_bucket[node_id] = bucket_index;

        if (previous_bucket_index == infinity_bucket_index) {
            infinity_bucket.erase(node_id);
        } else {
            buckets[previous_bucket_index].erase(node_id);
        }

        if (bucket_index >= buckets.size()) {
            buckets.resize(bucket_index * 2 + 1);
        }

        buckets[bucket_index].emplace(node_id);
    }

    /**
     * @brief Marks the node as active
     * @param node_id The local node id
     */
    void mark_as_active(const node_id_type node_id) {
        active_nodes.emplace(node_id);
    }

    /**
     * @brief Checks if there is an unprocessed node from the current bucket upwards.
     *		Disregards all nodes that are still 'infinitely far away'.
     * @param current_bucket The bucket from where to check
     * @return True iff there are nodes left on this rank
     */
    [[nodiscard]] bool has_nodes_left(const std::size_t current_bucket) const noexcept {
        if (current_bucket >= buckets.size()) {
            // There are no nodes left in this bucket and the larger ones, so this rank is finished.
            // We don't check infinity_bucket because of potentially unreachable nodes.
            return false;
        }

        for (auto bucket_index = current_bucket; bucket_index < buckets.size(); bucket_index++) {
            if (!buckets[bucket_index].empty()) {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Returns a view of all buckets starting at a given index
     * @param current_bucket Where to start
     * @return The view
     */
    [[nodiscard]] std::span<const std::unordered_set<node_id_type>> get_buckets(const std::size_t current_bucket) const noexcept {
        if (buckets.size() <= current_bucket) {
            return {};
        }

        return { buckets.data() + current_bucket, buckets.size() - current_bucket };
    }

    /**
     * @brief Returns a constant reference to the infinity bucket (which might be empty)
     * @return All nodes that are still 'infinitely far away' from the root of the SSSP
     */
    [[nodiscard]] const std::unordered_set<node_id_type>& get_nodes_at_infinity() const noexcept {
        return infinity_bucket;
    }

    /**
     * @brief Returns the current bucket the local node is in
     * @param node_id The id of the node
     * @return The index of the bucket. Can be infinity_bucket_index to represent the infinity bucket.
     */
    [[nodiscard]] std::size_t get_current_bucket(const node_id_type node_id) const noexcept {
        return node_to_bucket[node_id];
    }

private:
    std::vector<std::size_t> node_to_bucket{};

    std::vector<std::unordered_set<node_id_type>> buckets{};
    std::unordered_set<node_id_type> infinity_bucket{};

    std::unordered_set<node_id_type> active_nodes{};
};
