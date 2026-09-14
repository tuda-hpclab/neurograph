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

#include <compare>

template <typename d_type = distance_type>
struct VertexDistance {
    mpi_rank_type mpi_rank{};
    node_id_type node_id{};
    d_type distance{};

    [[nodiscard]] friend constexpr auto operator<=>(const VertexDistance<d_type>&, const VertexDistance<d_type>&) = default;
};

/**
 * @brief Orders VertexDistance objects by their distance only. Used as the comparator of the
 *		std::priority_queue in Dijkstra so that the vertex with the smallest distance is popped first.
 *		The defaulted operator<=> above must not be used for that purpose, as it would order by
 *		mpi_rank first and distance last.
 */
template <typename d_type = distance_type>
struct GreaterVertexDistance {
    [[nodiscard]] constexpr bool operator()(const VertexDistance<d_type>& lhs, const VertexDistance<d_type>& rhs) const noexcept {
        return lhs.distance > rhs.distance;
    }
};
