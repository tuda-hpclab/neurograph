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

#include "utility/Vec3.h"

#include <cpp-utility/hash/pair.hpp>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct LoadedNodes {
    std::vector<Vec3d> positions{};
    std::vector<std::string> area_names{};
    std::vector<std::string> signal_types{};
    std::vector<node_id_type> area_names_ind{};
    std::vector<node_id_type> signal_types_ind{};
};

using LoadedArcs = std::vector<std::unordered_map<std::pair<mpi_rank_type, node_id_type>, weight_type, utility::hash<std::pair<mpi_rank_type, node_id_type>>>>;
