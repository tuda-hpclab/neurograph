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

#include <cpp-utility/hash/hash.hpp>
#include <cpp-utility/hash/pair.hpp>

#include <string>
#include <unordered_map>
#include <utility>

class DistributedGraph;

class AreaConnectivity {
public:
    /// Maps an ordered pair of areas (each identified by its owning rank and local area id) to the summed arc weight between them.
    using AreaConnectivityIdMap = std::unordered_map<std::pair<NodeIdentifier, NodeIdentifier>, weight_type, utility::hash<std::pair<NodeIdentifier, NodeIdentifier>>>;

    /**
     * @brief For each ordered pair of areas (A, B), sums the weight of all arcs connecting a node in area A
     *		with a node in area B.
     *		Must be called on every MPI rank, returns the correct result only on the root rank.
     *		Complexity is O(A + R), possibly O(A + log R).
     * @param graph The distributed graph
     * @return A map from a pair of area names (source, target) to the summed arc weight between them.
     *		The full map is returned only on the root rank; every other rank returns an empty map.
     */
    [[nodiscard]] static AreaConnectivityMap compute_area_connectivity_strength(const DistributedGraph& graph);
};
