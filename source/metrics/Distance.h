#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "utility/PoolAllocator.h"

#include <algorithm>
#include <compare>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

template <typename d_type = distance_type>
struct VertexDistance {
    mpi_rank_type mpi_rank{};
    node_id_type node_id{};
    d_type distance{};

    [[nodiscard]] friend constexpr auto operator<=>(const VertexDistance<d_type>&, const VertexDistance<d_type>&) = default;
};

class NodePath {
public:
    NodePath() = default;

    explicit NodePath(const std::size_t reserved_capacity) {
        nodes.reserve(reserved_capacity);
    }

    void append_node(const global_node_id_type node) {
        nodes.emplace_back(node);
    }

    [[nodiscard]] std::span<const global_node_id_type> get_nodes() const noexcept {
        return nodes;
    }

private:
    std::vector<global_node_id_type, PoolAllocator<global_node_id_type>> nodes{};
};

struct VertexDistancePath {
    mpi_rank_type mpi_rank{};
    node_id_type node_id{};
    distance_type distance{};
    NodePath path{};
};

struct GreaterDistance {
    [[nodiscard]] bool operator()(const VertexDistancePath& lhs, const VertexDistancePath& rhs) const {
        return lhs.distance > rhs.distance;
    }
};

template <typename d_type = distance_type>
struct sssp_result {
    d_type sum_shortest_paths{};
    d_type maximum_finite_distance{};
    double sum_efficiency{};
    global_node_id_type number_unreachable_nodes{};
};

[[nodiscard]] inline sssp_result<distance_type> summarize_distances(const std::span<const distance_type> distances) {
    constexpr auto max_distance = std::numeric_limits<distance_type>::max();

    auto sum_efficiency = 0.0;
    auto sum_shortest_path_from_node = distance_type{ 0 };
    auto maximum_distance = distance_type{ 0 };
    auto number_unreachables_from_node = global_node_id_type{ 0 };

    for (const auto distance : distances) {
        if (distance == max_distance) {
            ++number_unreachables_from_node;
        } else {
            sum_shortest_path_from_node += distance;
            if (distance != 0) {
                sum_efficiency += (1.0 / static_cast<double>(distance));
            }
            maximum_distance = std::max(maximum_distance, distance);
        }
    }

    return { sum_shortest_path_from_node, maximum_distance, sum_efficiency, number_unreachables_from_node };
}

[[nodiscard]] inline sssp_result<inverse_distance_type> summarize_distances(const std::span<const inverse_distance_type> distances) {
    constexpr auto max_distance = std::numeric_limits<inverse_distance_type>::max();

    auto sum_efficiency = 0.0;
    auto sum_shortest_path_from_node = inverse_distance_type{ 0 };
    auto maximum_distance = inverse_distance_type{ 0 };
    auto number_unreachables_from_node = global_node_id_type{ 0 };

    for (const auto distance : distances) {
        if (distance == max_distance) {
            ++number_unreachables_from_node;
        } else {
            sum_shortest_path_from_node += distance;
            if (distance != 0) {
                sum_efficiency += (1.0 / static_cast<double>(distance));
            }
            maximum_distance = std::max(maximum_distance, distance);
        }
    }

    return { sum_shortest_path_from_node, maximum_distance, sum_efficiency, number_unreachables_from_node };
}
