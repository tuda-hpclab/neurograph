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

#include "graph/Arc.h"
#include "graph/GraphTypes.h"
#include "utility/DataCache.h"
#include "utility/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/RMAWindow.h"

#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief This class offers a distributed way to access the graph. Each MPI rank loads its own graph with its local files/arcs,
 *		which then connect via MPI and offer the functionality to access the other parts.
 */
class DistributedGraph {
public:
    /**
     * @brief Constructs a graph with the given data
     * @param nodes The nodes
     * @param in_arcs The in arcs
     * @param out_arcs The out arcs
     */
    [[nodiscard]] static DistributedGraph construct_graph(LoadedNodes nodes, const LoadedArcs& in_arcs, const LoadedArcs& out_arcs);

    /**
     * @brief Constructs a graph and loads the files in the specified directory.
     * @param path The directory that contains the filed
     * @param remove_self_arcs True iff all self arcs should be removed
     * @param undirected True iff all arcs should be considered as undirected
     * @param one_weight True iff all arcs should have weight one
     * @param file_prefix The prefix for the files
     */
    [[nodiscard]] static DistributedGraph construct_graph(const std::filesystem::path& path, bool remove_self_arcs, bool undirected, bool one_weight,
                                                          const std::string& file_prefix);

    /**
     * @brief Returns the position of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's position
     */
    [[nodiscard]] Vec3d get_node_position(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto position = nodes_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return position;
    }

    /**
     * @brief Returns the area id (local to the rank) of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's area id
     */
    [[nodiscard]] node_id_type get_node_area_localID(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto area_name_ind = area_names_ind_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return area_name_ind;
    }

    /**
     * @brief Returns the signal type id (local to the rank) of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's signal type id
     */
    [[nodiscard]] node_id_type get_node_signal_localID(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto signal_type_ind = signal_types_ind_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return signal_type_ind;
    }

    /**
     * @brief Returns the number of in arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's number of in arcs
     */
    [[nodiscard]] arc_id_type get_number_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto number_in_arcs = number_in_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return number_in_arcs;
    }

    /**
     * @brief Returns the number of out arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's number of out arcs
     */
    [[nodiscard]] arc_id_type get_number_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto number_out_arcs = number_out_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return number_out_arcs;
    }

    /**
     * @brief Returns the weight of all in arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's weight of in arcs
     */
    [[nodiscard]] weight_type get_weight_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto weight_in_arcs = weight_in_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return weight_in_arcs;
    }

    /**
     * @brief Returns the weight of all out arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's weight of out arcs
     */
    [[nodiscard]] weight_type get_weight_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto weight_out_arcs = weight_out_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return weight_out_arcs;
    }

    /**
     * @brief Returns the prefix of the in arcs in the RMA window of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's in-arc prefix
     */
    [[nodiscard]] arc_id_type get_prefix_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto prefix_in_arcs = prefix_in_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return prefix_in_arcs;
    }

    /**
     * @brief Returns the prefix of the out arcs in the RMA window of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's out-arc prefix
     */
    [[nodiscard]] arc_id_type get_prefix_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto prefix_out_arcs = prefix_out_arcs_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return prefix_out_arcs;
    }

    /**
     * @brief Returns the in arc of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @param arc_id The arc id of the node
     * @return The in arc
     */
    [[nodiscard]] InArc get_in_arc(const mpi_rank_type mpi_rank, const node_id_type node_id, const arc_id_type arc_id) const {
        const auto prefix_in_arc = get_prefix_in_arcs(mpi_rank, node_id);
        const auto arc = in_arcs_window.get(prefix_in_arc + arc_id, mpiPP::MPIRank{ mpi_rank });
        return arc;
    }

    /**
     * @brief Returns the in arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's in arcs
     */
    [[nodiscard]] std::span<const InArc> get_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        if (mpi_rank == mpiPP::MPIInfo::get_my_rank().get_rank()) {
            const auto prefix = get_prefix_in_arcs(mpi_rank, node_id);
            const auto size = get_number_in_arcs(mpi_rank, node_id);
            const auto span = std::span<const InArc>(in_arcs_window.get_pointer() + prefix, size);

            return span;
        }

        if (!in_arc_cache.contains(mpi_rank, node_id)) {
            const auto prefix_in_arc = get_prefix_in_arcs(mpi_rank, node_id);
            const auto number_in_arcs = get_number_in_arcs(mpi_rank, node_id);

            auto arcs = std::vector<InArc>(number_in_arcs);
            if (number_in_arcs > 0) {
                in_arcs_window.get(arcs.data(), number_in_arcs, prefix_in_arc, mpiPP::MPIRank{ mpi_rank });
            }

            in_arc_cache.insert(mpi_rank, node_id, std::move(arcs));
        }

        return in_arc_cache.get_value(mpi_rank, node_id);
    }

    /**
     * @brief Returns the out arc of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @param arc_id The arc id of the node
     * @return The out arc
     */
    [[nodiscard]] OutArc get_out_arc(const mpi_rank_type mpi_rank, const node_id_type node_id, const arc_id_type arc_id) const {
        const auto prefix_out_arc = get_prefix_out_arcs(mpi_rank, node_id);
        const auto arc = out_arcs_window.get(prefix_out_arc + arc_id, mpiPP::MPIRank{ mpi_rank });
        return arc;
    }

    /**
     * @brief Returns the out arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's out arcs
     */
    [[nodiscard]] std::span<const OutArc> get_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        if (mpi_rank == mpiPP::MPIInfo::get_my_rank().get_rank()) {
            const auto prefix = get_prefix_out_arcs(mpi_rank, node_id);
            const auto size = get_number_out_arcs(mpi_rank, node_id);
            const auto span = std::span<const OutArc>(out_arcs_window.get_pointer() + prefix, size);

            return span;
        }

        if (!out_arc_cache.contains(mpi_rank, node_id)) {
            const auto prefix_out_arc = get_prefix_out_arcs(mpi_rank, node_id);
            const auto number_out_arcs = get_number_out_arcs(mpi_rank, node_id);

            auto arcs = std::vector<OutArc>(number_out_arcs);
            if (!arcs.empty()) {
                out_arcs_window.get(arcs.data(), number_out_arcs, prefix_out_arc, mpiPP::MPIRank{ mpi_rank });
            }

            out_arc_cache.insert(mpi_rank, node_id, std::move(arcs));
        }

        return out_arc_cache.get_value(mpi_rank, node_id);
    }

    /**
     * @brief Returns the number of nodes on the current MPI rank
     * @return The number of local nodes
     */
    [[nodiscard]] node_id_type get_number_local_nodes() const noexcept {
        return local_number_nodes;
    }

    /**
     * @brief Returns the number of in arcs that end on the current MPI rank
     * @return The number of local in arcs
     */
    [[nodiscard]] arc_id_type get_number_local_in_arcs() const noexcept {
        return local_number_in_arcs;
    }

    /**
     * @brief Returns the distribution of the number of local in arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has k in arcs
     */
    [[nodiscard]] std::span<const arc_id_type> get_number_local_in_arc_distribution() const {
        return { number_in_arcs_window.get_pointer(), number_in_arcs_window.get_local_size() };
    }

    /**
     * @brief Returns the distribution of the weight of local in arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has the weight of k in arcs
     */
    [[nodiscard]] std::span<const weight_type> get_weight_in_arc_distribution() const {
        return { weight_in_arcs_window.get_pointer(), weight_in_arcs_window.get_local_size() };
    }

    /**
     * @brief Returns the number of out arcs that end on the current MPI rank
     * @return The number of local out arcs
     */
    [[nodiscard]] arc_id_type get_number_local_out_arcs() const noexcept {
        return local_number_out_arcs;
    }

    /**
     * @brief Returns the distribution of the number of local out arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has k out arcs
     */
    [[nodiscard]] std::span<const arc_id_type> get_number_local_out_arc_distribution() const {
        return { number_out_arcs_window.get_pointer(), number_out_arcs_window.get_local_size() };
    }

    /**
     * @brief Returns the distribution of the weight of local out arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has the weight of k out arcs
     */
    [[nodiscard]] std::span<const weight_type> get_weight_out_arc_distribution() const {
        return { weight_out_arcs_window.get_pointer(), weight_out_arcs_window.get_local_size() };
    }

    /**
     * @brief Returns the local area names (indexed by the local area-name ids)
     * @return The local area names as strings
     */
    [[nodiscard]] std::span<const std::string> get_local_area_names() const noexcept {
        return area_names;
    }

    /**
     * @brief Returns the local signal types (indexed by the local signal-type ids)
     * @return The local signal types as strings
     */
    [[nodiscard]] std::span<const std::string> get_local_signal_types() const noexcept {
        return signal_types;
    }

    /**
     * @brief Locks all rma windows as shared
     */
    void lock_all_rma_windows();

    /**
     * @brief Unlocks all rma windows
     */
    void unlock_all_rma_windows();

private:
    /**
     * @brief Constructs a new graph with the defined sizes.
     *		This is private to ensure that the graph is only constructed via the static methods.
     * @param number_nodes The number of local nodes
     * @param number_in_arcs The number of local in arcs
     * @param number_out_arcs The number of local out arcs
     */
    DistributedGraph(node_id_type number_nodes, arc_id_type number_in_arcs, arc_id_type number_out_arcs);

    void upload_nodes(LoadedNodes nodes);

    void upload_in_arcs(const LoadedArcs& in_arcs);

    void upload_out_arcs(const LoadedArcs& out_arcs);

    node_id_type local_number_nodes{};
    arc_id_type local_number_in_arcs{};
    arc_id_type local_number_out_arcs{};

    mpiPP::RMAWindow<Vec3d> nodes_window;

    mpiPP::RMAWindow<node_id_type> area_names_ind_window;
    mpiPP::RMAWindow<node_id_type> signal_types_ind_window;
    std::vector<std::string> area_names;
    std::vector<std::string> signal_types;

    mpiPP::RMAWindow<InArc> in_arcs_window;
    mpiPP::RMAWindow<arc_id_type> prefix_in_arcs_window;
    mpiPP::RMAWindow<arc_id_type> number_in_arcs_window;
    mpiPP::RMAWindow<weight_type> weight_in_arcs_window;

    mpiPP::RMAWindow<OutArc> out_arcs_window;
    mpiPP::RMAWindow<arc_id_type> prefix_out_arcs_window;
    mpiPP::RMAWindow<arc_id_type> number_out_arcs_window;
    mpiPP::RMAWindow<weight_type> weight_out_arcs_window;

    mutable DataCache<InArc> in_arc_cache{};
    mutable DataCache<OutArc> out_arc_cache{};
};
