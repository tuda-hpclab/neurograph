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

#include "GraphTypes.h"
#include "Types.h"

#include <filesystem>
#include <span>
#include <string_view>

/**
 * @brief This class offers the IO-capabilities for all loading from files.
 */
class FileLoader {
public:
    /**
     * @brief Uses the parent directory, the number of MPI ranks, and the current MPI rank to determine the path for the file that stores the nodes.
     * @param directory The parent directory
     * @param my_rank The current MPI rank
     * @param number_ranks The number of MPI ranks
     * @return The path to the file with the nodes
     */
    [[nodiscard]] static std::filesystem::path get_node_path(const std::filesystem::path& directory, mpi_rank_type my_rank, mpi_rank_type number_ranks);

    /**
     * @brief Uses the parent directory, a potential prefix (e.g., the number of steps), the number of MPI ranks,
     *		and the current MPI rank to determine the path for the file that stores the in arcs.
     * @param directory The parent directory
     * @param prefix The prefix, can be empty
     * @param my_rank The current MPI rank
     * @param number_ranks The number of MPI ranks
     * @return The path to the file with the in arcs
     */
    [[nodiscard]] static std::filesystem::path get_in_arcs_path(const std::filesystem::path& directory, std::string_view prefix, mpi_rank_type my_rank,
                                                                mpi_rank_type number_ranks);

    /**
     * @brief Uses the parent directory, a potential prefix (e.g., the number of steps), the number of MPI ranks,
     *		and the current MPI rank to determine the path for the file that stores the out arcs.
     * @param directory The parent directory
     * @param prefix The prefix, can be empty
     * @param my_rank The current MPI rank
     * @param number_ranks The number of MPI ranks
     * @return The path to the file with the out arcs
     */
    [[nodiscard]] static std::filesystem::path get_out_arcs_path(const std::filesystem::path& directory, std::string_view prefix, mpi_rank_type my_rank,
                                                                 mpi_rank_type number_ranks);

    /**
     * @brief Loads the nodes from the file
     * @param positions_file The file
     * @param my_rank The current MPI rank
     * @return A collection of all loaded data
     */
    [[nodiscard]] static LoadedNodes load_nodes(const std::filesystem::path& positions_file, mpi_rank_type my_rank);

    /**
     * @brief Loads the in arcs from the file. Checks that all arcs end on the current MPI rank and that the sources match the node distribution
     * @param in_network_file The file
     * @param my_rank The current MPI rank
     * @param node_distribution The number of nodes per MPI rank
     * @return A collection of all loaded in arcs
     */
    [[nodiscard]] static LoadedArcs load_in_arcs(const std::filesystem::path& in_network_file, mpi_rank_type my_rank,
                                                 std::span<const node_id_type> node_distribution);

    /**
     * @brief Loads the out arcs from the file. Checks that all arcs start on the current MPI rank and that the targets match the node distribution
     * @param out_network_file The file
     * @param my_rank The current MPI rank
     * @param node_distribution The number of nodes per MPI rank
     * @return A collection of all loaded out arcs
     */
    [[nodiscard]] static LoadedArcs load_out_arcs(const std::filesystem::path& out_network_file, mpi_rank_type my_rank,
                                                  std::span<const node_id_type> node_distribution);
};
