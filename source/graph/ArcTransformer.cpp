/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ArcTransformer.h"

#include "Types.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <ranges>
#include <utility>

ArcTransformer::Arcs ArcTransformer::remove_self_arcs(Arcs arcs, const mpi_rank_type my_rank) {
    for (auto node_id = node_id_type{ 0 }; node_id < arcs.size(); node_id++) {
        auto& arcs_from_node = arcs[node_id];
        arcs_from_node.erase({ my_rank, node_id });
    }

    return arcs;
}

ArcTransformer::Arcs ArcTransformer::remove_weight_zero(Arcs arcs) {
    for (auto& arcs_from_node : arcs) {
        for (auto it = arcs_from_node.begin(); it != arcs_from_node.end();) {
            if (it->second == 0) {
                it = arcs_from_node.erase(it);
            } else {
                ++it;
            }
        }
    }

    return arcs;
}

ArcTransformer::Arcs ArcTransformer::all_weights_positive(Arcs arcs) {
    std::ranges::for_each(
        arcs, [](auto& arcs_from_node) { std::ranges::for_each(arcs_from_node | std::ranges::views::values, [](auto& value) { value = std::abs(value); }); });

    return arcs;
}

ArcTransformer::Arcs ArcTransformer::all_weights_one(Arcs arcs) {
    std::ranges::for_each(arcs, [](auto& arcs_from_node) { std::ranges::fill(arcs_from_node | std::ranges::views::values, weight_type{ 1 }); });
    return arcs;
}

ArcTransformer::Arcs ArcTransformer::synchronize_arcs(Arcs incoming_arcs, Arcs outgoing_arcs) {
    for (auto target_id = std::size_t{ 0 }; target_id < incoming_arcs.size(); target_id++) {
        for (const auto& [key, weight] : incoming_arcs[target_id]) {
            const auto& [source_rank, source_id] = key;
            auto& current_weight = outgoing_arcs[target_id][std::pair{ source_rank, source_id }];
            current_weight += weight;
            if (current_weight == 0) {
                current_weight = 1;
            }
        }
    }

    return outgoing_arcs;
}
