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

#include <cpp-utility/hash/tuple.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

using mpi_rank_type = int;

using arc_id_type = std::uint32_t;
using node_id_type = std::uint32_t;
using global_node_id_type = std::uint64_t;

using weight_type = std::int32_t;
using distance_type = std::uint64_t;
using inverse_distance_type = double;

enum class DeltaSteppingEpochType : std::uint8_t {
    PushBruteForce,
    PushCheck,
    PushShortLong,
    PushInnerShort,
    PullModel,
};

// Below are more complex types which are declared here so that they are accessible in the MPIWrapper
struct AreaConnectivityInfo {
    mpi_rank_type source_rank{};
    node_id_type source_area_local_id{};
    mpi_rank_type target_rank{};
    node_id_type target_area_local_id{};
    weight_type weight{};
};

struct NodeIdentifier {
    mpi_rank_type owning_mpi_rank{};
    node_id_type node_id{};

    NodeIdentifier() = default;

    NodeIdentifier(const mpi_rank_type rank, const node_id_type node) noexcept {
        owning_mpi_rank = rank;
        node_id = node;
    }

    NodeIdentifier(const std::pair<mpi_rank_type, node_id_type>& pair) noexcept {
        owning_mpi_rank = pair.first;
        node_id = pair.second;
    }

    NodeIdentifier(const std::tuple<mpi_rank_type, node_id_type>& tuple) noexcept {
        owning_mpi_rank = std::get<0>(tuple);
        node_id = std::get<1>(tuple);
    }

    [[nodiscard]] bool has_uninitialized_rank() const noexcept {
        return owning_mpi_rank == std::numeric_limits<mpi_rank_type>::max();
    }

    [[nodiscard]] bool operator==(const NodeIdentifier& other) const noexcept = default;

    [[nodiscard]] static NodeIdentifier uninitialized() noexcept {
        return NodeIdentifier{ std::numeric_limits<mpi_rank_type>::max(), std::numeric_limits<node_id_type>::max() };
    }

    [[nodiscard]] operator std::pair<mpi_rank_type, node_id_type>() const noexcept {
        return { owning_mpi_rank, node_id };
    }

    [[nodiscard]] operator std::tuple<mpi_rank_type, node_id_type>() const noexcept {
        return { owning_mpi_rank, node_id };
    }

    friend std::ostream& operator<<(std::ostream& out, const NodeIdentifier& ni) {
        const auto& [owning_mpi_rank, node_id] = ni;
        out << '(' << owning_mpi_rank << ", " << node_id << ')';
        return out;
    }
};

namespace std {
template <>
struct hash<NodeIdentifier> {
    [[nodiscard]] std::size_t operator()(const NodeIdentifier& value) const {
        const auto tup = static_cast<std::tuple<mpi_rank_type, node_id_type>>(value);
        const auto hasher = utility::hash<std::tuple<mpi_rank_type, node_id_type>>{};
        return hasher(tup);
    }
};
} // namespace std

struct MinMax {
    arc_id_type min{ std::numeric_limits<arc_id_type>::max() };
    arc_id_type max{ std::numeric_limits<arc_id_type>::min() };
};

using AreaConnectivityMap = std::map<std::pair<std::string, std::string>, weight_type>;

template <typename d_type = distance_type>
struct apsp_global_result {
    double average_shortest_path_length{};
    double average_efficiency{};
    d_type diameter{};
    global_node_id_type number_disconnected_pairs{};
    std::unordered_map<global_node_id_type, global_node_id_type> cluster_sizes{};
};
