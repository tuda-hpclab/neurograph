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

#include "cpp-utility/hash/hash.hpp"
#include "cpp-utility/hash/pair.hpp"

#include <string>
#include <unordered_map>
#include <utility>

class DistributedGraph;

class AreaConnectivity {
public:
    using AreaIDConnecMap = std::unordered_map<std::pair<NodeIdentifier, NodeIdentifier>, weight_type, utility::hash<std::pair<NodeIdentifier, NodeIdentifier>>>;

    /**
     * Foreach combination of areas A and B, the function sums the weight
     * of all arcs connecting a node in area A with a node in area B.
     *
     * Parameters
     * graph:           A DistributedGraph (function is MPI compliant)
     *
     * Returns: std::unordered_map with the a pair of area names as key
     *          and the summed weight as value.
     *          (std::pair(area_name_A,area_name_B)->summed_weight)
     *
     * MPI Constraints: 	Function must be called by all ranks simultaneously
     * 						Function returns correct data only to rank stated in
     * 						the parameter result_rank
     */
    [[nodiscard]] static AreaConnectivityMap compute_area_connectivity_strength(const DistributedGraph& graph);
};
