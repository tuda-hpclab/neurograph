/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
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

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPICollectives.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPISynchronization.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

DistributedGraph DistributedGraph::construct_graph(LoadedNodes nodes, const LoadedArcs& in_arcs, const LoadedArcs& out_arcs) {
    const auto number_nodes = utility::save_cast<node_id_type>(nodes.positions.size());

    auto local_number_in_arcs = arc_id_type{ 0 };
    for (const auto& map : in_arcs) {
        local_number_in_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto local_number_out_arcs = arc_id_type{ 0 };
    for (const auto& map : out_arcs) {
        local_number_out_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto graph = DistributedGraph(number_nodes, local_number_in_arcs, local_number_out_arcs);
    graph.upload_nodes(std::move(nodes));
    graph.upload_in_arcs(in_arcs);
    graph.upload_out_arcs(out_arcs);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    graph.in_arc_cache.init(number_ranks);
    graph.out_arc_cache.init(number_ranks);

    return graph;
}

DistributedGraph DistributedGraph::construct_graph(const std::filesystem::path& path, bool remove_self_arcs, bool undirected, bool one_weight,
                                                   const std::string& file_prefix) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto node_path = FileLoader::get_node_path(path, my_rank, number_ranks);
    auto loaded_positions = FileLoader::load_nodes(node_path, my_rank);

    const auto number_nodes = utility::save_cast<node_id_type>(loaded_positions.positions.size());
    const auto& node_distribution = mpiPP::MPICollectives::all_gather(number_nodes);

    const auto in_arcs_path = FileLoader::get_in_arcs_path(path, file_prefix, my_rank, number_ranks);
    auto in_arcs = FileLoader::load_in_arcs(in_arcs_path, my_rank, node_distribution);

    const auto out_arcs_path = FileLoader::get_out_arcs_path(path, file_prefix, my_rank, number_ranks);
    auto out_arcs = FileLoader::load_out_arcs(out_arcs_path, my_rank, node_distribution);

    auto local_number_in_arcs = arc_id_type{ 0 };
    for (const auto& map : in_arcs) {
        local_number_in_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto local_number_out_arcs = arc_id_type{ 0 };
    for (const auto& map : out_arcs) {
        local_number_out_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto graph = DistributedGraph(number_nodes, local_number_in_arcs, local_number_out_arcs);

    graph.upload_nodes(std::move(loaded_positions));

    if (remove_self_arcs) {
        in_arcs = ArcTransformer::remove_self_arcs(std::move(in_arcs), my_rank);
        out_arcs = ArcTransformer::remove_self_arcs(std::move(out_arcs), my_rank);
    }

    if (undirected) {
        in_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));

        if (one_weight) {
            in_arcs = ArcTransformer::all_weights_one(std::move(in_arcs));
        }

        graph.upload_in_arcs(in_arcs);
        graph.upload_out_arcs(in_arcs); // This is no mistake, synchronizing the arcs makes both equal
    } else {
        if (one_weight) {
            in_arcs = ArcTransformer::all_weights_one(std::move(in_arcs));
            out_arcs = ArcTransformer::all_weights_one(std::move(out_arcs));
        }

        graph.upload_in_arcs(in_arcs);
        graph.upload_out_arcs(out_arcs);
    }

    graph.in_arc_cache.init(number_ranks);
    graph.out_arc_cache.init(number_ranks);

    return graph;
}

DistributedGraph::DistributedGraph(const node_id_type number_nodes, const arc_id_type number_in_arcs, const arc_id_type number_out_arcs)
    : nodes_window{ number_nodes }
    , area_names_ind_window{ number_nodes }
    , signal_types_ind_window{ number_nodes }
    , in_arcs_window{ number_in_arcs }
    , prefix_in_arcs_window{ number_nodes }
    , number_in_arcs_window{ number_nodes }
    , weight_in_arcs_window{ number_nodes }
    , out_arcs_window{ number_out_arcs }
    , prefix_out_arcs_window{ number_nodes }
    , number_out_arcs_window{ number_nodes }
    , weight_out_arcs_window{ number_nodes } {
}

void DistributedGraph::upload_nodes(LoadedNodes nodes) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    local_number_nodes = utility::save_cast<node_id_type>(nodes.positions.size());

    nodes_window.put(nodes.positions.data(), local_number_nodes, 0U, my_rank);
    area_names_ind_window.put(nodes.area_names_ind.data(), local_number_nodes, 0U, my_rank);
    signal_types_ind_window.put(nodes.signal_types_ind.data(), local_number_nodes, 0U, my_rank);

    area_names = std::move(nodes.area_names);
    signal_types = std::move(nodes.signal_types);
}

void DistributedGraph::upload_in_arcs(const LoadedArcs& in_arcs) {
    local_number_in_arcs = arc_id_type{ 0 };
    for (const auto& map : in_arcs) {
        local_number_in_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto number_in_arcs = std::vector<arc_id_type>(local_number_nodes);
    auto prefix_number_in_arcs = std::vector<arc_id_type>(local_number_nodes, 0);
    auto vector_weights = std::vector<weight_type>(local_number_nodes, 0);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    auto vector = std::vector<InArc>{};

    auto current_filling = std::size_t{ 0 };
    for (auto target_id = node_id_type{ 0 }; target_id < local_number_nodes && local_number_in_arcs > 0; ++target_id) {
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
            number_in_arcs[target_id] = utility::save_cast<node_id_type>(added_size);
            vector_weights[target_id] = sum_weights;
        }

        if (target_id > 0) {
            prefix_number_in_arcs[target_id] = prefix_number_in_arcs[target_id - 1] + utility::save_cast<arc_id_type>(in_arcs[target_id - 1].size());
        }

        vector.clear();
    }

    number_in_arcs_window.put(number_in_arcs.data(), local_number_nodes, 0U, my_rank);
    prefix_in_arcs_window.put(prefix_number_in_arcs.data(), local_number_nodes, 0U, my_rank);
    weight_in_arcs_window.put(vector_weights.data(), local_number_nodes, 0U, my_rank);
}

void DistributedGraph::upload_out_arcs(const LoadedArcs& out_arcs) {
    local_number_out_arcs = arc_id_type{ 0 };
    for (const auto& map : out_arcs) {
        local_number_out_arcs += utility::save_cast<arc_id_type>(map.size());
    }

    auto number_out_arcs = std::vector<arc_id_type>(local_number_nodes, 0);
    auto prefix_number_out_arcs = std::vector<arc_id_type>(local_number_nodes, 0);
    auto vector_weights = std::vector<weight_type>(local_number_nodes, 0);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    auto vector = std::vector<OutArc>{};

    auto current_filling = std::size_t{ 0 };
    for (auto source_id = node_id_type{ 0 }; source_id < local_number_nodes && local_number_out_arcs > 0; ++source_id) {
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
            number_out_arcs[source_id] = utility::save_cast<node_id_type>(added_size);
            vector_weights[source_id] = sum_weights;
        }

        if (source_id > 0) {
            prefix_number_out_arcs[source_id] = prefix_number_out_arcs[source_id - 1] + utility::save_cast<arc_id_type>(out_arcs[source_id - 1].size());
        }

        vector.clear();
    }

    number_out_arcs_window.put(number_out_arcs.data(), local_number_nodes, 0U, my_rank);
    prefix_out_arcs_window.put(prefix_number_out_arcs.data(), local_number_nodes, 0U, my_rank);
    weight_out_arcs_window.put(vector_weights.data(), local_number_nodes, 0U, my_rank);
}

void DistributedGraph::lock_all_rma_windows() {
    nodes_window.lock_each_rank();
    in_arcs_window.lock_each_rank();
    prefix_in_arcs_window.lock_each_rank();
    number_in_arcs_window.lock_each_rank();
    out_arcs_window.lock_each_rank();
    prefix_out_arcs_window.lock_each_rank();
    number_out_arcs_window.lock_each_rank();
    area_names_ind_window.lock_each_rank();
    signal_types_ind_window.lock_each_rank();

    mpiPP::MPISynchronization::barrier();
}

void DistributedGraph::unlock_all_rma_windows() {
    nodes_window.unlock_each_rank();
    in_arcs_window.unlock_each_rank();
    prefix_in_arcs_window.unlock_each_rank();
    number_in_arcs_window.unlock_each_rank();
    out_arcs_window.unlock_each_rank();
    prefix_out_arcs_window.unlock_each_rank();
    number_out_arcs_window.unlock_each_rank();
    area_names_ind_window.unlock_each_rank();
    signal_types_ind_window.unlock_each_rank();

    mpiPP::MPISynchronization::barrier();
}
