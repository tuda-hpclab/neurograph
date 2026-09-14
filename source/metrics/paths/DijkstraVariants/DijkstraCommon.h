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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

/**
 * @brief Helpers shared by all Dijkstra vertex queue variants.
 */
class DijkstraCommon {
public:
    /**
     * @brief Transforms an arc weight into the arc's length under the metric selected by d_type:
     *          - distance_type: the absolute weight
     *          - inverse_distance_type: 1/|weight|; zero-weight arcs map to the maximum representable
     *            value, i.e. they are infinitely long and must not be relaxed by the caller
     * @tparam d_type The type for the distances, either distance_type or inverse_distance_type
     * @param weight The weight of the arc
     * @return The length of the arc
     */
    template <typename d_type>
    [[nodiscard]] static d_type transform_weight(const weight_type weight) {
        static_assert(std::is_same_v<d_type, distance_type> || std::is_same_v<d_type, inverse_distance_type>,
                      "DijkstraCommon::transform_weight only supports distance_type and inverse_distance_type");

        // std::abs(weight) is undefined behavior for the most negative weight_type value,
        // so widen to a larger signed type before taking the absolute value.
        const auto absolute_weight = std::abs(static_cast<std::int64_t>(weight));

        if constexpr (std::is_same_v<d_type, distance_type>) {
            return utility::safe_cast<d_type>(absolute_weight);
        } else {
            if (absolute_weight == 0) {
                // A zero-weight arc carries no strength and is therefore infinitely long in the inverse metric.
                return std::numeric_limits<inverse_distance_type>::max();
            }
            return 1.0 / static_cast<inverse_distance_type>(absolute_weight);
        }
    }
};
