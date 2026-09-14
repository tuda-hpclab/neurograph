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

#include <algorithm>
#include <limits>
#include <span>

template <typename d_type = distance_type>
struct SsspResult {
    d_type sum_shortest_paths{};
    d_type maximum_finite_distance{};
    double sum_efficiency{};
    global_node_id_type number_unreachable_nodes{};
};

/**
 * @brief Summarizes the distances of a single SSSP run into aggregate values.
 *		Entries equal to the maximum representable value of d_type are treated as unreachable.
 * @tparam d_type The type of the distances, either distance_type or inverse_distance_type
 * @param distances The distances from a single source to every (global) node
 * @note The sum of distances is accumulated in d_type itself and can overflow silently for very
 *		large graphs with large weights; callers accumulate these sums further and inherit that limit.
 * @note Nodes at distance 0 are excluded from the efficiency sum, as their efficiency would be
 *		infinite; besides the source itself this affects nodes reached via zero-weight arcs.
 * @return A quadruple of
 *		(1) the sum of all finite distances,
 *		(2) the maximum finite distance (i.e., the eccentricity of the source),
 *		(3) the sum of the efficiencies (1 / distance) of all reachable, non-self nodes,
 *		(4) the number of unreachable nodes
 */
template <typename d_type = distance_type>
[[nodiscard]] SsspResult<d_type> summarize_distances(const std::span<const d_type> distances) {
    constexpr auto max_distance = std::numeric_limits<d_type>::max();

    auto sum_efficiency = 0.0;
    auto sum_shortest_path_from_node = d_type{ 0 };
    auto maximum_distance = d_type{ 0 };
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
