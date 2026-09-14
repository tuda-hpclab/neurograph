/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AreaConnectivity.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/data/prefix_sum.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <mpi-wrapper/collectives/MPIGatherV.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>
#include <mpi-wrapper/patterns/comm_patterns/Questions.h>

#include <climits>
#include <cstddef>
#include <ranges>

AreaConnectivityMap AreaConnectivity::compute_area_connectivity_strength(const DistributedGraph& graph) {
    auto timer_build_map = utility::RegionTimer{ "area-connectivity-build-map" };

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

    auto local_area_connectivity_map = AreaConnectivityIdMap{};

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto& node_connections = area_connections.get_answers_of_questioner_node(node_id);

        for (const auto& [source_rank, source_area_local_id, target_rank, target_area_local_id, weight] : node_connections) {
            auto source_identifier = NodeIdentifier(source_rank, source_area_local_id);
            auto target_identifier = NodeIdentifier(target_rank, target_area_local_id);
            local_area_connectivity_map[{ source_identifier, target_identifier }] += weight;
        }
    }

    timer_build_map.stop();

    auto timer_gather_local_maps = utility::RegionTimer{ "area-connectivity-gather-local-maps" };

    // Gather local area connection maps to result rank
    using AreaConnectivityIdEntry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;

    auto local_connectivity_entries = std::vector<AreaConnectivityIdEntry>();
    local_connectivity_entries.reserve(local_area_connectivity_map.size());
    for (const auto& [areas, weight] : local_area_connectivity_map) {
        local_connectivity_entries.emplace_back(areas.first.owning_mpi_rank, areas.first.node_id, areas.second.owning_mpi_rank, areas.second.node_id, weight);
    }

    const auto gathered_connectivity_entries = mpiPP::MPICollectives::gatherv(std::move(local_connectivity_entries), mpiPP::MPIRank::root_rank());

    timer_gather_local_maps.stop();

    auto timer_gather_area_names = utility::RegionTimer{ "area-connectivity-gather-area-names" };

    const auto local_area_names = graph.get_local_area_names();
    const auto number_local_area_names = local_area_names.size();

    auto local_area_names_length = std::vector<std::size_t>(number_local_area_names);
    std::ranges::transform(local_area_names, local_area_names_length.begin(), [](const std::string& str) { return str.length(); });

    const auto prefix_sum = utility::calculate_prefix_sum<std::size_t>(local_area_names_length);
    const auto total_length = number_local_area_names == 0
                                  ? std::size_t{ 0 }
                                  : local_area_names_length[number_local_area_names - 1] + prefix_sum[number_local_area_names - 1];

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
            const auto end = ((str_index + 1) == number_areas) ? flat.size() : prefixes[str_index + 1];

            const auto begin_iterator = flat.begin() + utility::safe_cast<std::ptrdiff_t>(begin);
            const auto end_iterator = flat.begin() + utility::safe_cast<std::ptrdiff_t>(end);

            names.emplace_back(begin_iterator, end_iterator);
        }
    }

    timer_gather_area_names.stop();

    auto timer_create_map = utility::RegionTimer{ "area-connectivity-create-map" };

    // Combine the gathered connectivity entries and translate the area ids to area names
    auto global_area_connectivity_map = AreaConnectivityMap{};

    for (const auto& connectivity_entries_of_rank : gathered_connectivity_entries) {
        for (const auto& [source_rank, source_area_id, target_rank, target_area_id, weight] : connectivity_entries_of_rank) {
            const auto& source_area = area_names[utility::safe_cast<std::size_t>(source_rank)][utility::safe_cast<std::size_t>(source_area_id)];
            const auto& target_area = area_names[utility::safe_cast<std::size_t>(target_rank)][utility::safe_cast<std::size_t>(target_area_id)];

            global_area_connectivity_map[{ source_area, target_area }] += utility::safe_cast<weight_type>(weight);
        }
    }

    timer_create_map.stop();

    mpiPP::MPIRegionTimerReport::reduce_and_print("AreaConnectivity");
    utility::RegionTimer::init();

    return global_area_connectivity_map;
}
