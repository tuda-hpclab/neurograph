/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GraphTest.h"

#include "Types.h"

#include "graph/ArcTransformer.h"
#include "graph/DistributedGraph.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {
struct ArcType {
    mpi_rank_type source_rank{};
    node_id_type source_node_id{};

    mpi_rank_type target_rank{};
    node_id_type target_node_id{};

    weight_type weight{};
    bool bidirectional{ false };
};

[[nodiscard]] std::pair<LoadedArcs, LoadedArcs> generate_arcs(const std::size_t size, const std::vector<ArcType>& Arcs,
                                                              const std::vector<std::size_t>& num_nodes_per_rank) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank_cast();
    auto in_arcs = LoadedArcs(size);
    auto out_arcs = LoadedArcs(size);

    constexpr auto convert_to_map_value_type = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
        return std::make_pair(std::make_pair(rank, node), weight);
    };

    for (const auto& [source_rank, source_node_id, target_rank, target_node_id, weight, bidirectional] : Arcs) {
        if (std::cmp_equal(source_rank, my_rank)) {
            if (source_node_id >= size) {
                spdlog::error("Trying to use non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {})", source_node_id, source_rank, source_rank,
                              source_node_id, target_rank, target_node_id, weight);
            }
            if (target_node_id >= num_nodes_per_rank[utility::safe_cast<std::size_t>(target_rank)]) {
                spdlog::error("Trying to use non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {})", target_node_id, target_rank, source_rank,
                              source_node_id, target_rank, target_node_id, weight);
            }
            if (source_node_id >= out_arcs.size()) {
                spdlog::error("Trying to insert arc of non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {}), out_arcs contains only {} nodes",
                              target_node_id, target_rank, source_rank, source_node_id, target_rank, target_node_id, weight, out_arcs.size());
            }
            out_arcs[source_node_id].insert(convert_to_map_value_type(target_rank, target_node_id, weight));
        }
        if (bidirectional && std::cmp_equal(target_rank, my_rank)) {
            if (source_node_id >= num_nodes_per_rank[utility::safe_cast<std::size_t>(source_rank)]) {
                spdlog::error("Trying to use non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {})", source_node_id, source_rank, source_rank,
                              source_node_id, target_rank, target_node_id, weight);
            }
            if (target_node_id >= size) {
                spdlog::error("Trying to use non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {})", target_node_id, target_rank, source_rank,
                              source_node_id, target_rank, target_node_id, weight);
            }
            if (target_node_id >= in_arcs.size()) {
                spdlog::error("Trying to insert arc of non-existent node_id {} in rank {}, arc: ({}, {}, {}, {}, {}), in_arcs contains only {} nodes",
                              target_node_id, target_rank, source_rank, source_node_id, target_rank, target_node_id, weight, in_arcs.size());
            }
            in_arcs[target_node_id].insert(convert_to_map_value_type(source_rank, source_node_id, weight));
        }
    }

    return { std::move(in_arcs), std::move(out_arcs) };
}
} // namespace

DistributedGraph GraphTest::get_small_four_rank_graph() {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto get_positions = [](const mpi_rank_type mpi_rank) -> std::vector<Vec3d> {
        switch (mpi_rank) {
        case 0: {
            auto positions_rank_0 = std::vector<Vec3d>{};
            positions_rank_0.emplace_back(0.1, 9.8, 4.7);
            positions_rank_0.emplace_back(9.6, 5.7, 1.4);
            return positions_rank_0;
        }
        case 1: {
            auto positions_rank_1 = std::vector<Vec3d>{};
            positions_rank_1.emplace_back(0.40, 0.31, 0.64);
            positions_rank_1.emplace_back(0.90, 0.83, 0.47);
            positions_rank_1.emplace_back(0.78, 0.45, 0.45);
            return positions_rank_1;
        }
        case 2: {
            auto positions_rank_2 = std::vector<Vec3d>{};
            positions_rank_2.emplace_back(0.18, 0.26, 1.00);
            positions_rank_2.emplace_back(0.72, 0.59, 0.06);
            return positions_rank_2;
        }
        case 3: {
            auto positions_rank_3 = std::vector<Vec3d>{};
            positions_rank_3.emplace_back(0.22, 0.29, 0.89);
            positions_rank_3.emplace_back(0.99, 0.51, 0.16);
            return positions_rank_3;
        }
        default:
            return {};
        }
    };

    const auto get_area_names = [](const mpi_rank_type mpi_rank) -> std::vector<std::string> {
        switch (mpi_rank) {
        case 0: {
            auto area_names_rank_0 = std::vector<std::string>{};
            area_names_rank_0.emplace_back("area_1");
            area_names_rank_0.emplace_back("area_2");
            return area_names_rank_0;
        }
        case 1: {
            auto area_names_rank_1 = std::vector<std::string>{};
            area_names_rank_1.emplace_back("area_1");
            area_names_rank_1.emplace_back("area_2");
            area_names_rank_1.emplace_back("area_4");
            return area_names_rank_1;
        }
        case 2: {
            auto area_names_rank_2 = std::vector<std::string>{};
            area_names_rank_2.emplace_back("area_1");
            area_names_rank_2.emplace_back("area_4");
            return area_names_rank_2;
        }
        case 3: {
            auto area_names_rank_3 = std::vector<std::string>{};
            area_names_rank_3.emplace_back("area_2");
            area_names_rank_3.emplace_back("area_4");
            return area_names_rank_3;
        }
        default:
            return {};
        }
    };

    const auto get_signal_types = [](const mpi_rank_type mpi_rank) -> std::vector<std::string> {
        switch (mpi_rank) {
        case 0: {
            auto signal_types_rank_0 = std::vector<std::string>{};
            signal_types_rank_0.emplace_back("excitatory");
            signal_types_rank_0.emplace_back("inhibitory");
            return signal_types_rank_0;
        }
        case 1: {
            auto signal_types_rank_1 = std::vector<std::string>{};
            signal_types_rank_1.emplace_back("inhibitory");
            signal_types_rank_1.emplace_back("excitatory");
            return signal_types_rank_1;
        }
        case 2: {
            auto signal_types_rank_2 = std::vector<std::string>{};
            signal_types_rank_2.emplace_back("excitatory");
            signal_types_rank_2.emplace_back("inhibitory");
            return signal_types_rank_2;
        }
        case 3: {
            auto signal_types_rank_3 = std::vector<std::string>{};
            signal_types_rank_3.emplace_back("excitatory");
            signal_types_rank_3.emplace_back("inhibitory");
            return signal_types_rank_3;
        }
        default:
            return {};
        }
    };

    const auto get_area_names_indices = [](const mpi_rank_type mpi_rank) -> std::vector<node_id_type> {
        switch (mpi_rank) {
        case 0: {
            auto area_names_indices_rank_0 = std::vector<node_id_type>{};
            area_names_indices_rank_0.emplace_back(0);
            area_names_indices_rank_0.emplace_back(1);
            return area_names_indices_rank_0;
        }
        case 1: {
            auto area_names_indices_rank_1 = std::vector<node_id_type>{};
            area_names_indices_rank_1.emplace_back(0);
            area_names_indices_rank_1.emplace_back(0);
            area_names_indices_rank_1.emplace_back(1);
            return area_names_indices_rank_1;
        }
        case 2: {
            auto area_names_indices_rank_2 = std::vector<node_id_type>{};
            area_names_indices_rank_2.emplace_back(0);
            area_names_indices_rank_2.emplace_back(1);
            return area_names_indices_rank_2;
        }
        case 3: {
            auto area_names_indices_rank_3 = std::vector<node_id_type>{};
            area_names_indices_rank_3.emplace_back(0);
            area_names_indices_rank_3.emplace_back(1);
            return area_names_indices_rank_3;
        }
        default:
            return {};
        }
    };

    const auto get_signal_types_indices = [](const mpi_rank_type mpi_rank) -> std::vector<node_id_type> {
        switch (mpi_rank) {
        case 0: {
            auto signal_types_indices_rank_0 = std::vector<node_id_type>{};
            signal_types_indices_rank_0.emplace_back(0);
            signal_types_indices_rank_0.emplace_back(0);
            return signal_types_indices_rank_0;
        }
        case 1: {
            auto signal_types_indices_rank_1 = std::vector<node_id_type>{};
            signal_types_indices_rank_1.emplace_back(1);
            signal_types_indices_rank_1.emplace_back(1);
            signal_types_indices_rank_1.emplace_back(1);
            return signal_types_indices_rank_1;
        }
        case 2: {
            auto signal_types_indices_rank_2 = std::vector<node_id_type>{};
            signal_types_indices_rank_2.emplace_back(0);
            signal_types_indices_rank_2.emplace_back(1);
            return signal_types_indices_rank_2;
        }
        case 3: {
            auto signal_types_indices_rank_3 = std::vector<node_id_type>{};
            signal_types_indices_rank_3.emplace_back(0);
            signal_types_indices_rank_3.emplace_back(1);
            return signal_types_indices_rank_3;
        }
        default:
            return {};
        }
    };

    const auto& positions = get_positions(my_rank);
    const auto& area_names = get_area_names(my_rank);
    const auto& signal_types = get_signal_types(my_rank);
    const auto& area_names_indices = get_area_names_indices(my_rank);
    const auto& signal_types_indices = get_signal_types_indices(my_rank);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
    auto num_nodes_per_rank = std::vector<std::size_t>{};
    num_nodes_per_rank.reserve(number_ranks);
    for (auto rank = 0U; rank < number_ranks; ++rank) {
        num_nodes_per_rank.push_back(get_positions(utility::safe_cast<mpi_rank_type>(rank)).size());
    }

    const auto num_nodes = num_nodes_per_rank[utility::safe_cast<std::size_t>(my_rank)];
    EXPECT_EQ(positions.size(), num_nodes);
    EXPECT_EQ(area_names.size(), num_nodes);
    EXPECT_EQ(area_names_indices.size(), num_nodes);
    EXPECT_EQ(signal_types_indices.size(), num_nodes);

    const auto Arcs = std::vector<ArcType>{
        { .source_rank = 0, .source_node_id = 0, .target_rank = 0, .target_node_id = 1, .weight = 100, .bidirectional = false },
        { .source_rank = 0, .source_node_id = 0, .target_rank = 1, .target_node_id = 0, .weight = 101, .bidirectional = false },
        { .source_rank = 0, .source_node_id = 1, .target_rank = 1, .target_node_id = 1, .weight = 102, .bidirectional = false },
        { .source_rank = 0, .source_node_id = 1, .target_rank = 3, .target_node_id = 1, .weight = 103, .bidirectional = false },

        { .source_rank = 1, .source_node_id = 0, .target_rank = 1, .target_node_id = 1, .weight = 104, .bidirectional = false },
        { .source_rank = 1, .source_node_id = 1, .target_rank = 1, .target_node_id = 1, .weight = 105, .bidirectional = false },
        { .source_rank = 1, .source_node_id = 2, .target_rank = 0, .target_node_id = 0, .weight = 106, .bidirectional = false },
        { .source_rank = 1, .source_node_id = 2, .target_rank = 0, .target_node_id = 1, .weight = 107, .bidirectional = false },

        { .source_rank = 2, .source_node_id = 0, .target_rank = 0, .target_node_id = 0, .weight = 108, .bidirectional = false },
        { .source_rank = 2, .source_node_id = 0, .target_rank = 2, .target_node_id = 0, .weight = 109, .bidirectional = false },
        { .source_rank = 2, .source_node_id = 1, .target_rank = 3, .target_node_id = 1, .weight = 110, .bidirectional = false },
    };

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };
    auto [in_arcs, out_arcs] = generate_arcs(positions.size(), Arcs, num_nodes_per_rank);

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_four_rank_graph() {
    const auto get_positions = [](const mpi_rank_type mpi_rank) {
        auto positions_rank_0 = std::vector<Vec3d>{};
        positions_rank_0.emplace_back(0.1, 9.8, 4.7);
        positions_rank_0.emplace_back(9.6, 5.7, 1.4);
        positions_rank_0.emplace_back(0.3, 3.9, 8.0);
        positions_rank_0.emplace_back(7.9, 2.8, 9.6);
        positions_rank_0.emplace_back(2.2, 4.9, 0.0);
        positions_rank_0.emplace_back(0.9, 5.2, 8.6);
        positions_rank_0.emplace_back(0.6, 5.8, 5.2);
        positions_rank_0.emplace_back(8.2, 2.0, 8.8);
        positions_rank_0.emplace_back(8.2, 4.8, 6.6);
        positions_rank_0.emplace_back(7.4, 4.4, 3.4);

        auto positions_rank_1 = std::vector<Vec3d>{};
        positions_rank_1.emplace_back(0.40, 0.31, 0.64);
        positions_rank_1.emplace_back(0.90, 0.83, 0.47);
        positions_rank_1.emplace_back(0.78, 0.45, 0.45);
        positions_rank_1.emplace_back(0.41, 0.22, 0.02);
        positions_rank_1.emplace_back(0.19, 0.93, 0.04);
        positions_rank_1.emplace_back(0.21, 0.28, 0.84);
        positions_rank_1.emplace_back(0.00, 0.41, 0.88);
        positions_rank_1.emplace_back(0.12, 0.55, 0.02);
        positions_rank_1.emplace_back(0.22, 0.91, 0.19);
        positions_rank_1.emplace_back(0.43, 0.99, 0.61);

        auto positions_rank_2 = std::vector<Vec3d>{};
        positions_rank_2.emplace_back(0.18, 0.26, 1.00);
        positions_rank_2.emplace_back(0.72, 0.59, 0.06);
        positions_rank_2.emplace_back(0.46, 0.67, 0.75);
        positions_rank_2.emplace_back(0.56, 0.44, 0.46);
        positions_rank_2.emplace_back(0.48, 0.12, 0.64);
        positions_rank_2.emplace_back(0.17, 0.79, 0.01);
        positions_rank_2.emplace_back(0.65, 0.17, 1.00);
        positions_rank_2.emplace_back(0.58, 0.87, 0.38);
        positions_rank_2.emplace_back(0.05, 0.57, 0.33);
        positions_rank_2.emplace_back(0.96, 0.44, 0.95);

        auto positions_rank_3 = std::vector<Vec3d>{};
        positions_rank_3.emplace_back(0.22, 0.29, 0.89);
        positions_rank_3.emplace_back(0.99, 0.51, 0.16);
        positions_rank_3.emplace_back(0.99, 0.72, 0.75);
        positions_rank_3.emplace_back(0.71, 0.01, 0.71);
        positions_rank_3.emplace_back(0.93, 0.68, 0.42);
        positions_rank_3.emplace_back(0.98, 0.47, 0.17);
        positions_rank_3.emplace_back(0.73, 0.48, 0.93);
        positions_rank_3.emplace_back(0.91, 0.35, 0.43);
        positions_rank_3.emplace_back(0.28, 0.12, 0.09);
        positions_rank_3.emplace_back(0.08, 0.67, 0.17);

        if (mpi_rank == 0) {
            return positions_rank_0;
        }

        if (mpi_rank == 1) {
            return positions_rank_1;
        }

        if (mpi_rank == 2) {
            return positions_rank_2;
        }

        return positions_rank_3;
    };

    const auto get_area_names = [](const mpi_rank_type mpi_rank) {
        auto area_names_rank_0 = std::vector<std::string>{};
        area_names_rank_0.emplace_back("area_1");
        area_names_rank_0.emplace_back("area_2");
        area_names_rank_0.emplace_back("area_3");

        auto area_names_rank_1 = std::vector<std::string>{};
        area_names_rank_1.emplace_back("area_1");
        area_names_rank_1.emplace_back("area_2");
        area_names_rank_1.emplace_back("area_4");

        auto area_names_rank_2 = std::vector<std::string>{};
        area_names_rank_2.emplace_back("area_1");
        area_names_rank_2.emplace_back("area_4");
        area_names_rank_2.emplace_back("area_5");

        auto area_names_rank_3 = std::vector<std::string>{};
        area_names_rank_3.emplace_back("area_2");
        area_names_rank_3.emplace_back("area_4");

        if (mpi_rank == 0) {
            return area_names_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_rank_2;
        }

        return area_names_rank_3;
    };

    const auto get_signal_types = [](const mpi_rank_type mpi_rank) {
        auto signal_types_rank_0 = std::vector<std::string>{};
        signal_types_rank_0.emplace_back("excitatory");
        signal_types_rank_0.emplace_back("inhibitory");

        auto signal_types_rank_1 = std::vector<std::string>{};
        signal_types_rank_1.emplace_back("inhibitory");
        signal_types_rank_1.emplace_back("excitatory");

        auto signal_types_rank_2 = std::vector<std::string>{};
        signal_types_rank_2.emplace_back("excitatory");
        signal_types_rank_2.emplace_back("inhibitory");

        auto signal_types_rank_3 = std::vector<std::string>{};
        signal_types_rank_3.emplace_back("excitatory");
        signal_types_rank_3.emplace_back("inhibitory");

        if (mpi_rank == 0) {
            return signal_types_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_rank_2;
        }

        return signal_types_rank_3;
    };

    const auto get_area_names_indices = [](const mpi_rank_type mpi_rank) {
        auto area_names_indices_rank_0 = std::vector<node_id_type>{};
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(2);

        auto area_names_indices_rank_1 = std::vector<node_id_type>{};
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(1);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(0);

        auto area_names_indices_rank_2 = std::vector<node_id_type>{};
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);

        auto area_names_indices_rank_3 = std::vector<node_id_type>{};
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);

        if (mpi_rank == 0) {
            return area_names_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_indices_rank_2;
        }

        return area_names_indices_rank_3;
    };

    const auto get_signal_types_indices = [](const mpi_rank_type mpi_rank) {
        auto signal_types_indices_rank_0 = std::vector<node_id_type>{};
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);

        auto signal_types_indices_rank_1 = std::vector<node_id_type>{};
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);

        auto signal_types_indices_rank_2 = std::vector<node_id_type>{};
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);

        auto signal_types_indices_rank_3 = std::vector<node_id_type>{};
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);

        if (mpi_rank == 0) {
            return signal_types_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_indices_rank_2;
        }

        return signal_types_indices_rank_3;
    };

    const auto get_in_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto in_arcs = LoadedArcs{};
        in_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2), ce(mpi_rank, 4, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 7, 1), ce(mpi_rank, 4, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 7, 2), ce(mpi_rank, 8, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 9, 1), ce(mpi_rank, 2, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 3, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 4, 2), ce(mpi_rank, 1, 3), ce(mpi_rank, 5, 9), ce(mpi_rank, 6, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 5, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 4, 1), ce(mpi_rank, 0, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 0, 1) });

        const auto other_rank = (mpi_rank + 3) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            in_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return in_arcs;
    };

    const auto get_out_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto out_arcs = LoadedArcs{};
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 6, 3) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 4, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 5, 2), ce(mpi_rank, 7, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 2, 2), ce(mpi_rank, 6, 2), ce(mpi_rank, 8, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 6, 9), ce(mpi_rank, 7, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 3, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 2), ce(mpi_rank, 4, 1) });

        const auto other_rank = (mpi_rank + 1) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            out_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return out_arcs;
    };

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto& positions = get_positions(my_rank);
    const auto& area_names = get_area_names(my_rank);
    const auto& signal_types = get_signal_types(my_rank);
    const auto& area_names_indices = get_area_names_indices(my_rank);
    const auto& signal_types_indices = get_signal_types_indices(my_rank);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    auto in_arcs = get_in_arcs(my_rank);
    auto out_arcs = get_out_arcs(my_rank);

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}

DistributedGraph GraphTest::get_standard_uu_four_rank_graph() {
    const auto get_positions = [](const mpi_rank_type mpi_rank) {
        auto positions_rank_0 = std::vector<Vec3d>{};
        positions_rank_0.emplace_back(0.1, 9.8, 4.7);
        positions_rank_0.emplace_back(9.6, 5.7, 1.4);
        positions_rank_0.emplace_back(0.3, 3.9, 8.0);
        positions_rank_0.emplace_back(7.9, 2.8, 9.6);
        positions_rank_0.emplace_back(2.2, 4.9, 0.0);
        positions_rank_0.emplace_back(0.9, 5.2, 8.6);
        positions_rank_0.emplace_back(0.6, 5.8, 5.2);
        positions_rank_0.emplace_back(8.2, 2.0, 8.8);
        positions_rank_0.emplace_back(8.2, 4.8, 6.6);
        positions_rank_0.emplace_back(7.4, 4.4, 3.4);

        auto positions_rank_1 = std::vector<Vec3d>{};
        positions_rank_1.emplace_back(0.40, 0.31, 0.64);
        positions_rank_1.emplace_back(0.90, 0.83, 0.47);
        positions_rank_1.emplace_back(0.78, 0.45, 0.45);
        positions_rank_1.emplace_back(0.41, 0.22, 0.02);
        positions_rank_1.emplace_back(0.19, 0.93, 0.04);
        positions_rank_1.emplace_back(0.21, 0.28, 0.84);
        positions_rank_1.emplace_back(0.00, 0.41, 0.88);
        positions_rank_1.emplace_back(0.12, 0.55, 0.02);
        positions_rank_1.emplace_back(0.22, 0.91, 0.19);
        positions_rank_1.emplace_back(0.43, 0.99, 0.61);

        auto positions_rank_2 = std::vector<Vec3d>{};
        positions_rank_2.emplace_back(0.18, 0.26, 1.00);
        positions_rank_2.emplace_back(0.72, 0.59, 0.06);
        positions_rank_2.emplace_back(0.46, 0.67, 0.75);
        positions_rank_2.emplace_back(0.56, 0.44, 0.46);
        positions_rank_2.emplace_back(0.48, 0.12, 0.64);
        positions_rank_2.emplace_back(0.17, 0.79, 0.01);
        positions_rank_2.emplace_back(0.65, 0.17, 1.00);
        positions_rank_2.emplace_back(0.58, 0.87, 0.38);
        positions_rank_2.emplace_back(0.05, 0.57, 0.33);
        positions_rank_2.emplace_back(0.96, 0.44, 0.95);

        auto positions_rank_3 = std::vector<Vec3d>{};
        positions_rank_3.emplace_back(0.22, 0.29, 0.89);
        positions_rank_3.emplace_back(0.99, 0.51, 0.16);
        positions_rank_3.emplace_back(0.99, 0.72, 0.75);
        positions_rank_3.emplace_back(0.71, 0.01, 0.71);
        positions_rank_3.emplace_back(0.93, 0.68, 0.42);
        positions_rank_3.emplace_back(0.98, 0.47, 0.17);
        positions_rank_3.emplace_back(0.73, 0.48, 0.93);
        positions_rank_3.emplace_back(0.91, 0.35, 0.43);
        positions_rank_3.emplace_back(0.28, 0.12, 0.09);
        positions_rank_3.emplace_back(0.08, 0.67, 0.17);

        if (mpi_rank == 0) {
            return positions_rank_0;
        }

        if (mpi_rank == 1) {
            return positions_rank_1;
        }

        if (mpi_rank == 2) {
            return positions_rank_2;
        }

        return positions_rank_3;
    };

    const auto get_area_names = [](const mpi_rank_type mpi_rank) {
        auto area_names_rank_0 = std::vector<std::string>{};
        area_names_rank_0.emplace_back("area_1");
        area_names_rank_0.emplace_back("area_2");
        area_names_rank_0.emplace_back("area_3");

        auto area_names_rank_1 = std::vector<std::string>{};
        area_names_rank_1.emplace_back("area_1");
        area_names_rank_1.emplace_back("area_2");
        area_names_rank_1.emplace_back("area_4");

        auto area_names_rank_2 = std::vector<std::string>{};
        area_names_rank_2.emplace_back("area_1");
        area_names_rank_2.emplace_back("area_4");
        area_names_rank_2.emplace_back("area_5");

        auto area_names_rank_3 = std::vector<std::string>{};
        area_names_rank_3.emplace_back("area_2");
        area_names_rank_3.emplace_back("area_4");

        if (mpi_rank == 0) {
            return area_names_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_rank_2;
        }

        return area_names_rank_3;
    };

    const auto get_signal_types = [](const mpi_rank_type mpi_rank) {
        auto signal_types_rank_0 = std::vector<std::string>{};
        signal_types_rank_0.emplace_back("excitatory");
        signal_types_rank_0.emplace_back("inhibitory");

        auto signal_types_rank_1 = std::vector<std::string>{};
        signal_types_rank_1.emplace_back("inhibitory");
        signal_types_rank_1.emplace_back("excitatory");

        auto signal_types_rank_2 = std::vector<std::string>{};
        signal_types_rank_2.emplace_back("excitatory");
        signal_types_rank_2.emplace_back("inhibitory");

        auto signal_types_rank_3 = std::vector<std::string>{};
        signal_types_rank_3.emplace_back("excitatory");
        signal_types_rank_3.emplace_back("inhibitory");

        if (mpi_rank == 0) {
            return signal_types_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_rank_2;
        }

        return signal_types_rank_3;
    };

    const auto get_area_names_indices = [](const mpi_rank_type mpi_rank) {
        auto area_names_indices_rank_0 = std::vector<node_id_type>{};
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(2);

        auto area_names_indices_rank_1 = std::vector<node_id_type>{};
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(1);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(0);

        auto area_names_indices_rank_2 = std::vector<node_id_type>{};
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);

        auto area_names_indices_rank_3 = std::vector<node_id_type>{};
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);

        if (mpi_rank == 0) {
            return area_names_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_indices_rank_2;
        }

        return area_names_indices_rank_3;
    };

    const auto get_signal_types_indices = [](const mpi_rank_type mpi_rank) {
        auto signal_types_indices_rank_0 = std::vector<node_id_type>{};
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);

        auto signal_types_indices_rank_1 = std::vector<node_id_type>{};
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);

        auto signal_types_indices_rank_2 = std::vector<node_id_type>{};
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);

        auto signal_types_indices_rank_3 = std::vector<node_id_type>{};
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);

        if (mpi_rank == 0) {
            return signal_types_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_indices_rank_2;
        }

        return signal_types_indices_rank_3;
    };

    const auto get_in_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto in_arcs = LoadedArcs{};
        in_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2), ce(mpi_rank, 4, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 7, 1), ce(mpi_rank, 4, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 7, 2), ce(mpi_rank, 8, 1) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 9, 1), ce(mpi_rank, 2, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 3, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 4, 2), ce(mpi_rank, 1, 3), ce(mpi_rank, 5, 9), ce(mpi_rank, 6, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 5, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 4, 1), ce(mpi_rank, 0, 2) });
        in_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 0, 1) });

        const auto other_rank = (mpi_rank + 3) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            in_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return in_arcs;
    };

    const auto get_out_arcs = [](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        constexpr auto ce = [](const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
            return std::make_pair(std::make_pair(rank, node), weight);
        };

        auto out_arcs = LoadedArcs{};
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 1), ce(mpi_rank, 8, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 0, 1), ce(mpi_rank, 6, 3) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 4, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 5, 2), ce(mpi_rank, 7, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 1), ce(mpi_rank, 2, 2), ce(mpi_rank, 6, 2), ce(mpi_rank, 8, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 5, 1), ce(mpi_rank, 6, 9), ce(mpi_rank, 7, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 6, 2), ce(mpi_rank, 9, 1) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 2, 1), ce(mpi_rank, 3, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 3, 1), ce(mpi_rank, 9, 2) });
        out_arcs.emplace_back(ac{ ce(mpi_rank, 1, 2), ce(mpi_rank, 4, 1) });

        const auto other_rank = (mpi_rank + 1) % 4;

        for (auto node_id = node_id_type{ 0 }; node_id < 10; ++node_id) {
            out_arcs[node_id][{ other_rank, node_id }] = 11;
        }

        return out_arcs;
    };

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto& positions = get_positions(my_rank);
    const auto& area_names = get_area_names(my_rank);
    const auto& signal_types = get_signal_types(my_rank);
    const auto& area_names_indices = get_area_names_indices(my_rank);
    const auto& signal_types_indices = get_signal_types_indices(my_rank);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    auto in_arcs = get_in_arcs(my_rank);
    auto out_arcs = get_out_arcs(my_rank);

    auto transformed_arcs = ArcTransformer::synchronize_arcs(std::move(in_arcs), std::move(out_arcs));
    auto clamped_arcs = ArcTransformer::all_weights_one(std::move(transformed_arcs));

    return DistributedGraph::construct_graph(std::move(loaded_nodes), clamped_arcs, clamped_arcs);
}

DistributedGraph GraphTest::get_full_four_rank_graph() {
    const auto get_number_nodes = [](const mpi_rank_type mpi_rank) -> node_id_type {
        if (mpi_rank == 0) {
            return 8;
        }

        if (mpi_rank == 1) {
            return 9;
        }

        if (mpi_rank == 2) {
            return 9;
        }

        return 10;
    };

    const auto get_positions = [](const mpi_rank_type mpi_rank) {
        auto positions_rank_0 = std::vector<Vec3d>{};
        positions_rank_0.emplace_back(0.1, 9.8, 4.7);
        positions_rank_0.emplace_back(9.6, 5.7, 1.4);
        positions_rank_0.emplace_back(0.3, 3.9, 8.0);
        positions_rank_0.emplace_back(7.9, 2.8, 9.6);
        positions_rank_0.emplace_back(2.2, 4.9, 0.0);
        positions_rank_0.emplace_back(0.9, 5.2, 8.6);
        positions_rank_0.emplace_back(0.6, 5.8, 5.2);
        positions_rank_0.emplace_back(8.2, 2.0, 8.8);

        auto positions_rank_1 = std::vector<Vec3d>{};
        positions_rank_1.emplace_back(0.40, 0.31, 0.64);
        positions_rank_1.emplace_back(0.90, 0.83, 0.47);
        positions_rank_1.emplace_back(0.78, 0.45, 0.45);
        positions_rank_1.emplace_back(0.41, 0.22, 0.02);
        positions_rank_1.emplace_back(0.19, 0.93, 0.04);
        positions_rank_1.emplace_back(0.21, 0.28, 0.84);
        positions_rank_1.emplace_back(0.00, 0.41, 0.88);
        positions_rank_1.emplace_back(0.12, 0.55, 0.02);
        positions_rank_1.emplace_back(0.22, 0.91, 0.19);

        auto positions_rank_2 = std::vector<Vec3d>{};
        positions_rank_2.emplace_back(0.18, 0.26, 1.00);
        positions_rank_2.emplace_back(0.72, 0.59, 0.06);
        positions_rank_2.emplace_back(0.46, 0.67, 0.75);
        positions_rank_2.emplace_back(0.56, 0.44, 0.46);
        positions_rank_2.emplace_back(0.48, 0.12, 0.64);
        positions_rank_2.emplace_back(0.17, 0.79, 0.01);
        positions_rank_2.emplace_back(0.65, 0.17, 1.00);
        positions_rank_2.emplace_back(0.58, 0.87, 0.38);
        positions_rank_2.emplace_back(0.05, 0.57, 0.33);

        auto positions_rank_3 = std::vector<Vec3d>{};
        positions_rank_3.emplace_back(0.22, 0.29, 0.89);
        positions_rank_3.emplace_back(0.99, 0.51, 0.16);
        positions_rank_3.emplace_back(0.99, 0.72, 0.75);
        positions_rank_3.emplace_back(0.71, 0.01, 0.71);
        positions_rank_3.emplace_back(0.93, 0.68, 0.42);
        positions_rank_3.emplace_back(0.98, 0.47, 0.17);
        positions_rank_3.emplace_back(0.73, 0.48, 0.93);
        positions_rank_3.emplace_back(0.91, 0.35, 0.43);
        positions_rank_3.emplace_back(0.28, 0.12, 0.09);
        positions_rank_3.emplace_back(0.08, 0.67, 0.17);

        if (mpi_rank == 0) {
            return positions_rank_0;
        }

        if (mpi_rank == 1) {
            return positions_rank_1;
        }

        if (mpi_rank == 2) {
            return positions_rank_2;
        }

        return positions_rank_3;
    };

    const auto get_area_names = [](const mpi_rank_type mpi_rank) {
        auto area_names_rank_0 = std::vector<std::string>{};
        area_names_rank_0.emplace_back("area_1");
        area_names_rank_0.emplace_back("area_2");
        area_names_rank_0.emplace_back("area_3");

        auto area_names_rank_1 = std::vector<std::string>{};
        area_names_rank_1.emplace_back("area_1");
        area_names_rank_1.emplace_back("area_2");
        area_names_rank_1.emplace_back("area_4");

        auto area_names_rank_2 = std::vector<std::string>{};
        area_names_rank_2.emplace_back("area_1");
        area_names_rank_2.emplace_back("area_4");
        area_names_rank_2.emplace_back("area_5");

        auto area_names_rank_3 = std::vector<std::string>{};
        area_names_rank_3.emplace_back("area_2");
        area_names_rank_3.emplace_back("area_4");

        if (mpi_rank == 0) {
            return area_names_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_rank_2;
        }

        return area_names_rank_3;
    };

    const auto get_signal_types = [](const mpi_rank_type mpi_rank) {
        auto signal_types_rank_0 = std::vector<std::string>{};
        signal_types_rank_0.emplace_back("excitatory");
        signal_types_rank_0.emplace_back("inhibitory");

        auto signal_types_rank_1 = std::vector<std::string>{};
        signal_types_rank_1.emplace_back("inhibitory");
        signal_types_rank_1.emplace_back("excitatory");

        auto signal_types_rank_2 = std::vector<std::string>{};
        signal_types_rank_2.emplace_back("excitatory");
        signal_types_rank_2.emplace_back("inhibitory");

        auto signal_types_rank_3 = std::vector<std::string>{};
        signal_types_rank_3.emplace_back("excitatory");
        signal_types_rank_3.emplace_back("inhibitory");

        if (mpi_rank == 0) {
            return signal_types_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_rank_2;
        }

        return signal_types_rank_3;
    };

    const auto get_area_names_indices = [](const mpi_rank_type mpi_rank) {
        auto area_names_indices_rank_0 = std::vector<node_id_type>{};
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(1);
        area_names_indices_rank_0.emplace_back(0);
        area_names_indices_rank_0.emplace_back(2);

        auto area_names_indices_rank_1 = std::vector<node_id_type>{};
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(1);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(0);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(2);
        area_names_indices_rank_1.emplace_back(0);

        auto area_names_indices_rank_2 = std::vector<node_id_type>{};
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(1);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(0);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);
        area_names_indices_rank_2.emplace_back(2);

        auto area_names_indices_rank_3 = std::vector<node_id_type>{};
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(0);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(1);
        area_names_indices_rank_3.emplace_back(0);

        if (mpi_rank == 0) {
            return area_names_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return area_names_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return area_names_indices_rank_2;
        }

        return area_names_indices_rank_3;
    };

    const auto get_signal_types_indices = [](const mpi_rank_type mpi_rank) {
        auto signal_types_indices_rank_0 = std::vector<node_id_type>{};
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);
        signal_types_indices_rank_0.emplace_back(1);
        signal_types_indices_rank_0.emplace_back(0);

        auto signal_types_indices_rank_1 = std::vector<node_id_type>{};
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(0);
        signal_types_indices_rank_1.emplace_back(1);
        signal_types_indices_rank_1.emplace_back(1);

        auto signal_types_indices_rank_2 = std::vector<node_id_type>{};
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(0);
        signal_types_indices_rank_2.emplace_back(1);
        signal_types_indices_rank_2.emplace_back(1);

        auto signal_types_indices_rank_3 = std::vector<node_id_type>{};
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(1);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(0);
        signal_types_indices_rank_3.emplace_back(1);

        if (mpi_rank == 0) {
            return signal_types_indices_rank_0;
        }

        if (mpi_rank == 1) {
            return signal_types_indices_rank_1;
        }

        if (mpi_rank == 2) {
            return signal_types_indices_rank_2;
        }

        return signal_types_indices_rank_3;
    };

    const auto get_in_arcs = [get_number_nodes](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        const auto number_local_nodes = get_number_nodes(mpi_rank);
        auto in_arcs = LoadedArcs{ number_local_nodes, ac{} };

        for (auto my_node_id = node_id_type{ 0 }; my_node_id < number_local_nodes; ++my_node_id) {
            for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
                const auto number_distant_nodes = get_number_nodes(rank);
                for (auto other_node_id = node_id_type{ 0 }; other_node_id < number_distant_nodes; ++other_node_id) {
                    if (mpiPP::MPIInfo::get_my_rank().get_rank() == rank && my_node_id == other_node_id) {
                        continue;
                    }

                    in_arcs[my_node_id][{ rank, other_node_id }] = 1;
                }
            }
        }

        return in_arcs;
    };

    const auto get_out_arcs = [get_number_nodes](const mpi_rank_type mpi_rank) {
        using ac = LoadedArcs::value_type;

        const auto number_local_nodes = get_number_nodes(mpi_rank);
        auto out_arcs = LoadedArcs{ number_local_nodes, ac{} };

        for (auto my_node_id = node_id_type{ 0 }; my_node_id < number_local_nodes; ++my_node_id) {
            for (auto rank = mpi_rank_type{ 0 }; rank < 4; ++rank) {
                const auto number_distant_nodes = get_number_nodes(rank);
                for (auto other_node_id = node_id_type{ 0 }; other_node_id < number_distant_nodes; ++other_node_id) {
                    if (mpiPP::MPIInfo::get_my_rank().get_rank() == rank && my_node_id == other_node_id) {
                        continue;
                    }

                    out_arcs[my_node_id][{ rank, other_node_id }] = 1;
                }
            }
        }

        return out_arcs;
    };

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto& positions = get_positions(my_rank);
    const auto& area_names = get_area_names(my_rank);
    const auto& signal_types = get_signal_types(my_rank);
    const auto& area_names_indices = get_area_names_indices(my_rank);
    const auto& signal_types_indices = get_signal_types_indices(my_rank);

    auto loaded_nodes = LoadedNodes{ positions, area_names, signal_types, area_names_indices, signal_types_indices };

    auto in_arcs = get_in_arcs(my_rank);
    auto out_arcs = get_out_arcs(my_rank);

    return DistributedGraph::construct_graph(std::move(loaded_nodes), in_arcs, out_arcs);
}
