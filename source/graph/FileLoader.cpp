/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FileLoader.h"

#include "Types.h"

#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/hash/pair.hpp>

#include <fmt/format.h>
#include <fmt/std.h>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
// Builds the zero-padded "rank_<id>" prefix that all per-rank files share, e.g. "rank_007" for rank 7 out of 1000.
[[nodiscard]] std::string make_rank_prefix(const mpi_rank_type my_rank, const mpi_rank_type number_ranks) {
    const auto number_of_digits_ranks = std::to_string(number_ranks - 1).length();
    const auto number_of_digits_my_rank = std::to_string(my_rank).length();

    const auto number_of_zeros = number_of_digits_ranks - number_of_digits_my_rank;
    const auto zeros = std::string(number_of_zeros, '0');

    return std::string("rank_") + zeros + std::to_string(my_rank);
}
} // namespace

std::filesystem::path FileLoader::get_node_path(const std::filesystem::path& directory, const mpi_rank_type my_rank, const mpi_rank_type number_ranks) {
    // build the rank prefix for the files
    const auto rank_prefix = make_rank_prefix(my_rank, number_ranks);

    // build the strings for the files
    const auto positions_file = rank_prefix + "_positions.txt";

    // build the new paths for files
    const auto positions_path = directory / "positions" / positions_file;
    utility::Exception::check(std::filesystem::exists(positions_path), "The positions for rank {} do not exist: {}", my_rank, positions_path);

    return positions_path;
}

std::filesystem::path FileLoader::get_in_arcs_path(const std::filesystem::path& directory, const std::string_view prefix, const mpi_rank_type my_rank,
                                                   const mpi_rank_type number_ranks) {
    // build the rank prefix for the files
    auto rank_prefix = make_rank_prefix(my_rank, number_ranks);

    // build the strings for the files
    const auto in_network_file = rank_prefix.append(prefix) + "_in_network.txt";

    // build the new paths for files
    const auto in_network_path = directory / "network" / in_network_file;
    utility::Exception::check(std::filesystem::exists(in_network_path), "The in arcs for rank {} do not exist: {}", my_rank, in_network_path);

    return in_network_path;
}

std::filesystem::path FileLoader::get_out_arcs_path(const std::filesystem::path& directory, const std::string_view prefix, const mpi_rank_type my_rank,
                                                    mpi_rank_type number_ranks) {
    // build the rank prefix for the files
    auto rank_prefix = make_rank_prefix(my_rank, number_ranks);

    // build the strings for the files
    const auto out_network_file = rank_prefix.append(prefix) + "_out_network.txt";

    // build the new paths for files
    const auto out_network_path = directory / "network" / out_network_file;
    utility::Exception::check(std::filesystem::exists(out_network_path), "The out arcs for rank {} do not exist: {}", my_rank, out_network_path);

    return out_network_path;
}

LoadedNodes FileLoader::load_nodes(const std::filesystem::path& positions_file, const mpi_rank_type my_rank) {
    utility::Exception::check(std::filesystem::exists(positions_file), "The positions for rank {} do not exist: {}", my_rank, positions_file);

    auto file = std::ifstream(positions_file);

    if (!file.good()) {
        utility::Exception::fail("File is not good: {}", positions_file);
    }

    auto positions = std::vector<Vec3d>{};

    auto area_names = std::vector<std::string>{};
    auto signal_types = std::vector<std::string>{};

    auto area_names_ind = std::vector<node_id_type>{};
    auto signal_types_ind = std::vector<node_id_type>{};

    auto area_names_set = std::unordered_map<std::string, std::size_t>{};
    auto signal_types_set = std::unordered_map<std::string, std::size_t>{};

    for (std::string line{}; std::getline(file, line);) {
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto iss = std::istringstream{ line };

        auto id = node_id_type{ 0 };

        auto pos_x = 0.0;
        auto pos_y = 0.0;
        auto pos_z = 0.0;

        auto area_name = std::string{};
        auto signal_type = std::string{};

        const auto success = (iss >> id) && (iss >> pos_x) && (iss >> pos_y) && (iss >> pos_z) && (iss >> signal_type);

        if (!success) {
            spdlog::warn("Error processing line: \"{}\"", line);
            continue;
        }

        const auto found_area_name = (iss >> area_name) && true;
        if (!found_area_name) {
            // Default value if no area name is provided
            area_name = "unknown";
        } else {
            // The area_name came before the signal_type
            std::swap(signal_type, area_name);
        }

        if (const auto expected_id = positions.size() + 1; expected_id != id) {
            spdlog::error("Expected to load node with id {} but loaded id {}", expected_id, id);
            continue;
        }

        positions.emplace_back(pos_x, pos_y, pos_z);

        if (const auto area_name_pos = area_names_set.find(area_name); area_name_pos != area_names_set.end()) {
            // Area name already encountered
            area_names_ind.emplace_back(utility::safe_cast<node_id_type>(area_name_pos->second));
        } else {
            // Area name new
            area_names_set.try_emplace(area_name, area_names.size());
            area_names_ind.emplace_back(utility::safe_cast<node_id_type>(area_names.size()));
            area_names.emplace_back(std::move(area_name));
        }

        if (const auto signal_type_pos = signal_types_set.find(signal_type); signal_type_pos != signal_types_set.end()) {
            // Signal name already encountered
            signal_types_ind.emplace_back(utility::safe_cast<node_id_type>(signal_type_pos->second));
        } else {
            // signal type new
            signal_types_set.try_emplace(signal_type, signal_types.size());
            signal_types_ind.emplace_back(utility::safe_cast<node_id_type>(signal_types.size()));
            signal_types.emplace_back(std::move(signal_type));
        }
    }

    return { std::move(positions), std::move(area_names), std::move(signal_types), std::move(area_names_ind), std::move(signal_types_ind) };
}

LoadedArcs FileLoader::load_in_arcs(const std::filesystem::path& in_network_file, const mpi_rank_type my_rank,
                                    const std::span<const node_id_type> node_distribution) {
    const auto local_number_nodes = node_distribution[utility::safe_cast<std::size_t>(my_rank)];

    auto in_arcs = LoadedArcs{ local_number_nodes };

    auto file = std::ifstream(in_network_file);

    if (!file.good()) {
        utility::Exception::fail("File is not good: {}", in_network_file);
    }

    for (std::string line{}; std::getline(file, line);) {
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto iss = std::istringstream{ line };

        auto target_rank = mpi_rank_type{};
        auto target_id = node_id_type{};

        auto source_rank = mpi_rank_type{};
        auto source_id = node_id_type{};

        auto weight = weight_type{};

        const auto success = (iss >> target_rank) && (iss >> target_id) && (iss >> source_rank) && (iss >> source_id) && (iss >> weight);
        if (!success) {
            spdlog::warn("Error processing line: \"{}\"", line);
            continue;
        }

        int plastic{};
        if (const auto plastic_exists = success && (iss >> plastic); plastic_exists && plastic == 0) {
            continue;
        }

        if (weight == 0) {
            spdlog::warn("The weight of a arc was 0: \"{}\"", line);
            continue;
        }

        if (target_rank != my_rank) {
            spdlog::warn("Loading in arcs for rank {} but found an in arc that is directed to rank {}", my_rank, target_rank);
            continue;
        }

        // IDs are 1-based
        --target_id;
        --source_id;

        if (target_id >= local_number_nodes) {
            spdlog::warn("Loaded an in arc with target id {} but I only have {} nodes.", target_id, local_number_nodes);
            continue;
        }

        // The source of an in arc lives on source_rank; make sure it exists in the node distribution.
        if (source_rank < mpi_rank_type{ 0 } || std::cmp_greater_equal(source_rank, node_distribution.size())) {
            spdlog::warn("Loaded an in arc from rank {} but there are only {} ranks.", source_rank, node_distribution.size());
            continue;
        }

        if (source_id >= node_distribution[utility::safe_cast<std::size_t>(source_rank)]) {
            spdlog::warn("Loaded an in arc from source id {} on rank {} but that rank only has {} nodes.", source_id, source_rank,
                         node_distribution[utility::safe_cast<std::size_t>(source_rank)]);
            continue;
        }

        in_arcs[target_id][std::pair{ source_rank, source_id }] += weight;
    }

    return in_arcs;
}

LoadedArcs FileLoader::load_out_arcs(const std::filesystem::path& out_network_file, const mpi_rank_type my_rank,
                                     const std::span<const node_id_type> node_distribution) {
    const auto local_number_nodes = node_distribution[utility::safe_cast<std::size_t>(my_rank)];
    auto out_arcs = LoadedArcs{ local_number_nodes };

    auto file = std::ifstream(out_network_file);

    if (!file.good()) {
        utility::Exception::fail("File is not good: {}", out_network_file);
    }

    for (std::string line{}; std::getline(file, line);) {
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto iss = std::istringstream{ line };

        auto target_rank = mpi_rank_type{};
        auto target_id = node_id_type{};

        auto source_rank = mpi_rank_type{};
        auto source_id = node_id_type{};

        auto weight = weight_type{};

        const auto success = (iss >> target_rank) && (iss >> target_id) && (iss >> source_rank) && (iss >> source_id) && (iss >> weight);

        if (!success) {
            spdlog::warn("Error processing line: \"{}\"", line);
            continue;
        }

        int plastic{};
        if (const auto plastic_exists = success && (iss >> plastic); plastic_exists && plastic == 0) {
            continue;
        }

        if (weight == 0) {
            spdlog::warn("The weight of a arc was 0: \"{}\"", line);
            continue;
        }

        if (source_rank != my_rank) {
            spdlog::warn("Loading out arcs for rank {} but found an out arc that is directed to rank {}", my_rank, source_rank);
            continue;
        }

        // IDs are 1-based
        --target_id;
        --source_id;

        if (source_id >= local_number_nodes) {
            spdlog::warn("Loaded an out arc with source id {} but I only have {} nodes.", source_id, local_number_nodes);
            continue;
        }

        // The target of an out arc lives on target_rank; make sure it exists in the node distribution.
        if (target_rank < mpi_rank_type{ 0 } || std::cmp_greater_equal(target_rank, node_distribution.size())) {
            spdlog::warn("Loaded an out arc to rank {} but there are only {} ranks.", target_rank, node_distribution.size());
            continue;
        }

        if (target_id >= node_distribution[utility::safe_cast<std::size_t>(target_rank)]) {
            spdlog::warn("Loaded an out arc to target id {} on rank {} but that rank only has {} nodes.", target_id, target_rank,
                         node_distribution[utility::safe_cast<std::size_t>(target_rank)]);
            continue;
        }

        out_arcs[source_id][std::pair{ target_rank, target_id }] += weight;
    }

    return out_arcs;
}
