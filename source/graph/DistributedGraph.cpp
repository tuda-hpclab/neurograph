/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "DistributedGraph.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/ArcTransformer.h"
#include "graph/FileLoader.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/collectives/MPIAllGather.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPISynchronization.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] arc_id_type count_arcs(const LoadedArcs& arcs) {
    auto number_arcs = arc_id_type{ 0 };
    for (const auto& map : arcs) {
        number_arcs += utility::safe_cast<arc_id_type>(map.size());
    }

    return number_arcs;
}
} // namespace

DistributedGraph DistributedGraph::construct_graph(LoadedNodes nodes, const LoadedArcs& in_arcs, const LoadedArcs& out_arcs) {
    const auto number_nodes = utility::safe_cast<node_id_type>(nodes.positions.size());

    const auto local_number_in_arcs = count_arcs(in_arcs);
    const auto local_number_out_arcs = count_arcs(out_arcs);

    auto graph = DistributedGraph(number_nodes, local_number_in_arcs, local_number_out_arcs);
    graph.upload_nodes(std::move(nodes));
    graph.upload_in_arcs(in_arcs);
    graph.upload_out_arcs(out_arcs);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    graph.in_arc_cache.init(number_ranks);
    graph.out_arc_cache.init(number_ranks);

    // Every rank only writes its own window contents, so no rank may read remotely before all of them finished uploading
    mpiPP::MPISynchronization::barrier();

    return graph;
}

DistributedGraph DistributedGraph::construct_graph(const std::filesystem::path& path, bool remove_self_arcs, bool undirected, bool one_weight,
                                                   const std::string& file_prefix) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto node_path = FileLoader::get_node_path(path, my_rank, number_ranks);
    auto loaded_positions = FileLoader::load_nodes(node_path, my_rank);

    const auto number_nodes = utility::safe_cast<node_id_type>(loaded_positions.positions.size());
    const auto& node_distribution = mpiPP::MPICollectives::all_gather(number_nodes);

    const auto in_arcs_path = FileLoader::get_in_arcs_path(path, file_prefix, my_rank, number_ranks);
    auto in_arcs = FileLoader::load_in_arcs(in_arcs_path, my_rank, node_distribution);

    const auto out_arcs_path = FileLoader::get_out_arcs_path(path, file_prefix, my_rank, number_ranks);
    auto out_arcs = FileLoader::load_out_arcs(out_arcs_path, my_rank, node_distribution);

    // The transformations must run before the RMA windows are sized: synchronize_arcs can produce more arcs than
    // were loaded (the union of in and out arcs), so counting the loaded arcs first would size the windows too small.
    if (remove_self_arcs) {
        in_arcs = ArcTransformer::remove_self_arcs(std::move(in_arcs), my_rank);
        out_arcs = ArcTransformer::remove_self_arcs(std::move(out_arcs), my_rank);
    }

    if (undirected) {
        in_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));

        if (one_weight) {
            in_arcs = ArcTransformer::all_weights_one(std::move(in_arcs));
        }
    } else if (one_weight) {
        in_arcs = ArcTransformer::all_weights_one(std::move(in_arcs));
        out_arcs = ArcTransformer::all_weights_one(std::move(out_arcs));
    }

    // Determine the arc counts from the transformed arcs so the RMA windows are sized correctly.
    const auto local_number_in_arcs = count_arcs(in_arcs);
    // synchronize_arcs stored the merged arcs in in_arcs and out_arcs was moved from; both windows use in_arcs.
    const auto local_number_out_arcs = undirected ? local_number_in_arcs : count_arcs(out_arcs);

    auto graph = DistributedGraph(number_nodes, local_number_in_arcs, local_number_out_arcs);

    graph.upload_nodes(std::move(loaded_positions));

    if (undirected) {
        graph.upload_in_arcs(in_arcs);
        graph.upload_out_arcs(in_arcs); // This is no mistake, synchronizing the arcs makes both equal
    } else {
        graph.upload_in_arcs(in_arcs);
        graph.upload_out_arcs(out_arcs);
    }

    graph.in_arc_cache.init(number_ranks);
    graph.out_arc_cache.init(number_ranks);

    // Every rank only writes its own window contents, so no rank may read remotely before all of them finished uploading
    mpiPP::MPISynchronization::barrier();

    return graph;
}

DistributedGraph::DistributedGraph(const node_id_type number_nodes, const arc_id_type number_in_arcs, const arc_id_type number_out_arcs)
    : nodes_window{ number_nodes }
    , area_names_ind_window{ number_nodes }
    , signal_types_ind_window{ number_nodes }
    , in_arcs_window{ number_in_arcs }
    , in_arc_info_window{ number_nodes }
    , weight_in_arcs_window{ number_nodes }
    , out_arcs_window{ number_out_arcs }
    , out_arc_info_window{ number_nodes }
    , weight_out_arcs_window{ number_nodes } {
}

void DistributedGraph::upload_nodes(LoadedNodes nodes) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    local_number_nodes = utility::safe_cast<node_id_type>(nodes.positions.size());

    nodes_window.put(nodes.positions.data(), local_number_nodes, 0U, my_rank);
    area_names_ind_window.put(nodes.area_names_ind.data(), local_number_nodes, 0U, my_rank);
    signal_types_ind_window.put(nodes.signal_types_ind.data(), local_number_nodes, 0U, my_rank);

    area_names = std::move(nodes.area_names);
    signal_types = std::move(nodes.signal_types);
}

void DistributedGraph::upload_in_arcs(const LoadedArcs& in_arcs) {
    local_number_in_arcs = count_arcs(in_arcs);

    auto in_arc_infos = std::vector<ArcInfo>(local_number_nodes);
    auto vector_weights = std::vector<weight_type>(local_number_nodes, 0);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    auto vector = std::vector<InArc>{};

    auto current_filling = std::size_t{ 0 };
    for (auto target_id = node_id_type{ 0 }; target_id < local_number_nodes && local_number_in_arcs > 0; ++target_id) {
        // The prefix of a node is the number of arcs actually stored before it, i.e., the current fill level.
        in_arc_infos[target_id].prefix_arcs = utility::safe_cast<arc_id_type>(current_filling);

        const auto& node_in_arcs = in_arcs[target_id];

        auto sum_weights = weight_type{ 0 };
        vector.reserve(node_in_arcs.size());
        for (const auto& [key, weight] : node_in_arcs) {
            if (weight == 0) {
                continue;
            }

            const auto& [source_rank, source_id] = key;
            vector.emplace_back(source_rank, source_id, weight);

            sum_weights += weight;
        }

        if (!vector.empty()) {
            const auto added_size = vector.size();
            in_arcs_window.put(vector.data(), added_size, current_filling, my_rank);

            current_filling += added_size;
            in_arc_infos[target_id].number_arcs = utility::safe_cast<arc_id_type>(added_size);
            vector_weights[target_id] = sum_weights;
        }

        vector.clear();
    }

    // Arcs with weight zero are skipped above, so the actually stored count can be smaller than the sum of map sizes.
    local_number_in_arcs = utility::safe_cast<arc_id_type>(current_filling);

    in_arc_info_window.put(in_arc_infos.data(), local_number_nodes, 0U, my_rank);
    weight_in_arcs_window.put(vector_weights.data(), local_number_nodes, 0U, my_rank);
}

void DistributedGraph::upload_out_arcs(const LoadedArcs& out_arcs) {
    local_number_out_arcs = count_arcs(out_arcs);

    auto out_arc_infos = std::vector<ArcInfo>(local_number_nodes);
    auto vector_weights = std::vector<weight_type>(local_number_nodes, 0);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    auto vector = std::vector<OutArc>{};

    auto current_filling = std::size_t{ 0 };
    for (auto source_id = node_id_type{ 0 }; source_id < local_number_nodes && local_number_out_arcs > 0; ++source_id) {
        // The prefix of a node is the number of arcs actually stored before it, i.e., the current fill level.
        out_arc_infos[source_id].prefix_arcs = utility::safe_cast<arc_id_type>(current_filling);

        const auto& node_out_arcs = out_arcs[source_id];

        auto sum_weights = weight_type{ 0 };
        vector.reserve(node_out_arcs.size());
        for (const auto& [key, weight] : node_out_arcs) {
            if (weight == 0) {
                continue;
            }

            const auto& [target_rank, target_id] = key;
            vector.emplace_back(target_rank, target_id, weight);

            sum_weights += weight;
        }

        if (!vector.empty()) {
            const auto added_size = vector.size();
            out_arcs_window.put(vector.data(), added_size, current_filling, my_rank);

            current_filling += added_size;
            out_arc_infos[source_id].number_arcs = utility::safe_cast<arc_id_type>(added_size);
            vector_weights[source_id] = sum_weights;
        }

        vector.clear();
    }

    // Arcs with weight zero are skipped above, so the actually stored count can be smaller than the sum of map sizes.
    local_number_out_arcs = utility::safe_cast<arc_id_type>(current_filling);

    out_arc_info_window.put(out_arc_infos.data(), local_number_nodes, 0U, my_rank);
    weight_out_arcs_window.put(vector_weights.data(), local_number_nodes, 0U, my_rank);
}

void DistributedGraph::lock_all_rma_windows() {
    nodes_window.lock_each_rank();
    in_arcs_window.lock_each_rank();
    in_arc_info_window.lock_each_rank();
    weight_in_arcs_window.lock_each_rank();
    out_arcs_window.lock_each_rank();
    out_arc_info_window.lock_each_rank();
    weight_out_arcs_window.lock_each_rank();
    area_names_ind_window.lock_each_rank();
    signal_types_ind_window.lock_each_rank();

    mpiPP::MPISynchronization::barrier();
}

void DistributedGraph::unlock_all_rma_windows() {
    nodes_window.unlock_each_rank();
    in_arcs_window.unlock_each_rank();
    in_arc_info_window.unlock_each_rank();
    weight_in_arcs_window.unlock_each_rank();
    out_arcs_window.unlock_each_rank();
    out_arc_info_window.unlock_each_rank();
    weight_out_arcs_window.unlock_each_rank();
    area_names_ind_window.unlock_each_rank();
    signal_types_ind_window.unlock_each_rank();

    mpiPP::MPISynchronization::barrier();
}
