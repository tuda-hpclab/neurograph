/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AreaConnectivity.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "utility/Timer.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/data/prefix_sum.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/comm_patterns/Questions.h"

#include <climits>
#include <ranges>

AreaConnectivityMap AreaConnectivity::compute_area_connectivity_strength(const DistributedGraph& graph) {
    auto timer = Timer{ "AreaConnectivity" };

    // Build local area connection map
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = graph.get_number_local_nodes();

    const auto transfer_connection_sources = [my_rank, &graph](const node_id_type node_id) {
        using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<AreaConnectivityInfo, node_id_type>::result_type;

        constexpr auto max_area_id = std::numeric_limits<node_id_type>::max();

        const auto out_arcs = graph.get_out_arcs(my_rank, node_id);
        const auto node_area_id = graph.get_node_area_localID(my_rank, node_id);
        auto connection_sources = ResultType(out_arcs.size());

        std::ranges::transform(out_arcs, connection_sources.begin(), [&my_rank, &node_area_id](const OutArc& out_arc) {
            return ResultType::value_type{ out_arc.target_rank, out_arc.target_id,
                                           AreaConnectivityInfo{ my_rank, node_area_id, out_arc.target_rank, max_area_id, out_arc.weight } };
        });

        return connection_sources;
    };

    const auto fill_connection_target_areaID = [my_rank, &graph](const node_id_type node_id, AreaConnectivityInfo connection) {
        connection.target_area_local_id = graph.get_node_area_localID(my_rank, node_id);
        return connection;
    };

    const auto area_connections = mpiPP::comm_patterns::node_to_node_question<AreaConnectivityInfo, AreaConnectivityInfo, node_id_type>(number_local_nodes, transfer_connection_sources, fill_connection_target_areaID);

    auto my_rank_connecID_map = AreaIDConnecMap{};

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto& node_connections = area_connections.get_answers_of_questioner_node(node_id);

        for (const auto& [source_rank, source_area_local_id, target_rank, target_area_local_id, weight] : node_connections) {
            auto source_nameID = NodeIdentifier(source_rank, source_area_local_id);
            auto target_nameID = NodeIdentifier(target_rank, target_area_local_id);
            my_rank_connecID_map[{ source_nameID, target_nameID }] += weight;
        }
    }

    timer.time_section("BuildAreaConnectMap");

    // Gather local area connection maps to result rank
    using connecID_map_data = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;

    auto connecID_list = std::vector<connecID_map_data>();
    for (const auto& [first, second] : my_rank_connecID_map) {
        connecID_list.emplace_back(first.first.owning_mpi_rank, first.first.node_id, first.second.owning_mpi_rank, first.second.node_id, second);
    }

    const auto ranks_to_connecID_data = mpiPP::MPICollectives::gatherv(std::move(connecID_list), mpiPP::MPIRank::root_rank());

    timer.time_section("GatherLocalAreaMaps");

    const auto local_area_names = graph.get_local_area_names();
    const auto number_local_area_names = local_area_names.size();

    auto local_area_names_length = std::vector<std::size_t>{};
    local_area_names_length.resize(number_local_area_names);
    std::ranges::transform(local_area_names, local_area_names_length.begin(), [](const std::string& str) { return str.length(); });

    const auto prefix_sum = utility::calculate_prefix_sum<std::size_t>(local_area_names_length);
    const auto total_length = local_area_names_length[number_local_area_names - 1] + prefix_sum[number_local_area_names - 1];

    auto flattened_area_names = std::vector<char>{};
    flattened_area_names.reserve(total_length);

    for (const auto& str : local_area_names) {
        flattened_area_names.insert(flattened_area_names.end(), str.begin(), str.end());
    }

    const auto area_names_prefixes = mpiPP::MPICollectives::gatherv(prefix_sum, mpiPP::MPIRank::root_rank());
    const auto area_names_flat = mpiPP::MPICollectives::gatherv(flattened_area_names, mpiPP::MPIRank::root_rank());

    auto area_names = std::vector<std::vector<std::string>>{};
    for (auto rank_cast = std::size_t{ 0 }; rank_cast < area_names_prefixes.size(); rank_cast++) {
        const auto& prefixes = area_names_prefixes[rank_cast];
        const auto& flat = area_names_flat[rank_cast];

        const auto number_areas = prefixes.size();

        auto& names = area_names.emplace_back();
        names.reserve(number_areas);

        for (auto str_index = std::size_t{ 0 }; str_index < number_areas; str_index++) {
            const auto begin = prefixes[str_index];
            const auto end = ((str_index + 1) == number_areas) ? area_names_flat[rank_cast].size() : prefixes[str_index + 1];

            const auto begin_iterator = flat.begin() + begin;
            const auto end_iterator = flat.begin() + end;

            names.emplace_back(begin_iterator, end_iterator);
        }
    }

    timer.time_section("GatherAreaNames");

    // Combine connecID_maps and transfer IDs to area names
    auto global_connec_name_map = AreaConnectivityMap{};

    for (const auto& connec_data_of_rank : ranks_to_connecID_data) {
        for (const auto& [source_rank, source_node_id, target_rank, target_node_id, weight] : connec_data_of_rank) {
            const auto& source_area = area_names[utility::save_cast<std::size_t>(source_rank)][utility::save_cast<std::size_t>(source_node_id)];
            const auto& target_area = area_names[utility::save_cast<std::size_t>(target_rank)][utility::save_cast<std::size_t>(target_node_id)];

            global_connec_name_map[{ source_area, target_area }] += utility::save_cast<weight_type>(weight);
        }
    }

    timer.time_section("CreateAreaConnectivityMap");
    timer.finish();

    return global_connec_name_map;
}
