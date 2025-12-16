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

#include "cpp-utility/Cast.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

/**
 * @brief This class offers a cache for data based on a unique identifier of MPI rank and node id.
 *		It caches a std::vector<DataType> which can be retrieved as a std::span<const DataType>.
 * @tparam DataType The type to cache.
 */
template <typename DataType>
class DataCache {
public:
    /**
     * @brief Initializes the cache to hold as many MPI ranks as given. Does not check the value.
     * @param number_ranks The number of MPI ranks
     */
    void init(const mpi_rank_type number_ranks) {
        cache.resize(utility::save_cast<std::size_t>(number_ranks));
    }

    /**
     * @brief Clears the cache for each MPI rank (but does not clear the number of MPI ranks).
     */
    void clear() {
        for (auto& entry : cache) {
            entry.clear();
        }
    }

    /**
     * @brief Checks if for a given MPI rank and a given node id, the cache has values for it.
     * @param rank The MPI rank
     * @param node_id The node id
     * @return True iff there are values for the node on the specified rank
     */
    [[nodiscard]] bool contains(const mpi_rank_type rank, const node_id_type node_id) {
        const auto& entry = cache[utility::save_cast<std::size_t>(rank)];
        const auto& where_it = entry.find(node_id);

        return where_it != entry.cend();
    }

    /**
     * @brief Returns the data for the given MPI rank and node id. Make sure it's already there!
     * @param rank The MPI rank
     * @param node_id The node id
     * @return A view of the data
     */
    [[nodiscard]] std::span<const DataType> get_value(const mpi_rank_type rank, const node_id_type node_id) {
        return cache[utility::save_cast<std::size_t>(rank)][node_id];
    }

    /**
     * @brief Returns the data for the given MPI rank and node id. Returns an empty optional if the data is not present.
     * @param rank The MPI rank
     * @param node_id The node id
     * @return An optional view of the data, might be empty
     */
    [[nodiscard]] std::optional<std::span<const DataType>> get_value_opt(const mpi_rank_type rank, const node_id_type node_id) {
        const auto& entry = cache[utility::save_cast<std::size_t>(rank)];
        const auto& where_it = entry.find(node_id);

        if (where_it == entry.cend()) {
            return {};
        }

        return where_it->second;
    }

    /**
     * @brief Inserts the given data for the MPI rank and node id. Overwrites already stored data, if present.
     * @param rank The MPI rank
     * @param node_id The node id
     * @param values The newly cached values
     */
    void insert(const mpi_rank_type rank, const node_id_type node_id, std::vector<DataType>&& values) {
        auto& entry = cache[utility::save_cast<std::size_t>(rank)];
        entry.emplace(node_id, std::move(values));
    }

private:
    std::vector<std::unordered_map<node_id_type, std::vector<DataType>>> cache{};
};
