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

#include <cpp-utility/memory/PoolAllocator.hpp>

#include <cstddef>
#include <span>
#include <vector>

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
    std::vector<global_node_id_type, utility::PoolAllocator<global_node_id_type>> nodes{};
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
