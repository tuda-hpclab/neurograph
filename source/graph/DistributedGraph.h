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

#include "graph/Arc.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/data-structure/DataCache.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/rma/RMAWindow.h>

#include <filesystem>
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief This struct summarizes the arcs of one node: how many arcs are stored for it and at which offset (prefix)
 *		those arcs start in the arc RMA window, i.e., they occupy [prefix_arcs, prefix_arcs + number_arcs).
 *		Both values are packed into eight bytes so that locating the arcs of a remote node costs a single
 *		remote memory access instead of two.
 */
struct ArcInfo {
    arc_id_type number_arcs{};
    arc_id_type prefix_arcs{};

    [[nodiscard]] friend auto operator<=>(const ArcInfo&, const ArcInfo&) = default;

    friend std::ostream& operator<<(std::ostream& out, const ArcInfo& arc_info) {
        out << '(' << arc_info.number_arcs << ", " << arc_info.prefix_arcs << ')';
        return out;
    }
};

static_assert(sizeof(ArcInfo) == 8, "ArcInfo must summarize the arcs of a node in eight bytes");

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
     * @brief Returns the number of in arcs of the specified node and their prefix in the in-arc RMA window.
     *		Might perform communication via MPI, but fetches both values with a single remote memory access.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's in-arc info
     */
    [[nodiscard]] ArcInfo get_in_arc_info(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto in_arc_info = in_arc_info_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return in_arc_info;
    }

    /**
     * @brief Returns the number of out arcs of the specified node and their prefix in the out-arc RMA window.
     *		Might perform communication via MPI, but fetches both values with a single remote memory access.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's out-arc info
     */
    [[nodiscard]] ArcInfo get_out_arc_info(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto out_arc_info = out_arc_info_window.get(node_id, mpiPP::MPIRank{ mpi_rank });
        return out_arc_info;
    }

    /**
     * @brief Returns the number of in arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's number of in arcs
     */
    [[nodiscard]] arc_id_type get_number_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto number_in_arcs = get_in_arc_info(mpi_rank, node_id).number_arcs;
        return number_in_arcs;
    }

    /**
     * @brief Returns the number of out arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's number of out arcs
     */
    [[nodiscard]] arc_id_type get_number_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto number_out_arcs = get_out_arc_info(mpi_rank, node_id).number_arcs;
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
        const auto prefix_in_arcs = get_in_arc_info(mpi_rank, node_id).prefix_arcs;
        return prefix_in_arcs;
    }

    /**
     * @brief Returns the prefix of the out arcs in the RMA window of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's out-arc prefix
     */
    [[nodiscard]] arc_id_type get_prefix_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        const auto prefix_out_arcs = get_out_arc_info(mpi_rank, node_id).prefix_arcs;
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
        const auto in_arc_info = get_in_arc_info(mpi_rank, node_id);
        const auto arc = in_arcs_window.get(in_arc_info.prefix_arcs + arc_id, mpiPP::MPIRank{ mpi_rank });
        return arc;
    }

    /**
     * @brief Returns the in arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's in arcs
     */
    [[nodiscard]] std::span<const InArc> get_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        // Only fetch the arc info if it is actually needed, i.e., if the arcs are not cached already
        if (mpi_rank != mpiPP::MPIInfo::get_my_rank().get_rank() && in_arc_cache.contains(mpi_rank, node_id)) {
            return in_arc_cache.get_value(mpi_rank, node_id);
        }

        return get_in_arcs(mpi_rank, node_id, get_in_arc_info(mpi_rank, node_id));
    }

    /**
     * @brief Returns the in arcs of the specified node, reusing an arc info the caller fetched before.
     *		Might perform communication via MPI, but saves the remote memory access for the arc info that
     *		the overload without it performs. Intended for callers that fetched the arc info anyway, e.g.,
     *		to compare the degrees of two nodes before downloading the arcs of one of them.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @param in_arc_info The in-arc info of the node, i.e., the result of get_in_arc_info for the same node
     * @return The node's in arcs
     */
    [[nodiscard]] std::span<const InArc> get_in_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id, const ArcInfo in_arc_info) const {
        if (mpi_rank == mpiPP::MPIInfo::get_my_rank().get_rank()) {
            const auto span = std::span<const InArc>(in_arcs_window.get_pointer() + in_arc_info.prefix_arcs, in_arc_info.number_arcs);

            return span;
        }

        if (!in_arc_cache.contains(mpi_rank, node_id)) {
            auto arcs = std::vector<InArc>(in_arc_info.number_arcs);
            if (in_arc_info.number_arcs > 0) {
                in_arcs_window.get(arcs.data(), in_arc_info.number_arcs, in_arc_info.prefix_arcs, mpiPP::MPIRank{ mpi_rank });
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
        const auto out_arc_info = get_out_arc_info(mpi_rank, node_id);
        const auto arc = out_arcs_window.get(out_arc_info.prefix_arcs + arc_id, mpiPP::MPIRank{ mpi_rank });
        return arc;
    }

    /**
     * @brief Returns the out arcs of the specified node. Might perform communication via MPI.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @return The node's out arcs
     */
    [[nodiscard]] std::span<const OutArc> get_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id) const {
        // Only fetch the arc info if it is actually needed, i.e., if the arcs are not cached already
        if (mpi_rank != mpiPP::MPIInfo::get_my_rank().get_rank() && out_arc_cache.contains(mpi_rank, node_id)) {
            return out_arc_cache.get_value(mpi_rank, node_id);
        }

        return get_out_arcs(mpi_rank, node_id, get_out_arc_info(mpi_rank, node_id));
    }

    /**
     * @brief Returns the out arcs of the specified node, reusing an arc info the caller fetched before.
     *		Might perform communication via MPI, but saves the remote memory access for the arc info that
     *		the overload without it performs. Intended for callers that fetched the arc info anyway, e.g.,
     *		to compare the degrees of two nodes before downloading the arcs of one of them.
     * @param mpi_rank The MPI rank (can be the current MPI rank)
     * @param node_id The node id on the MPI rank
     * @param out_arc_info The out-arc info of the node, i.e., the result of get_out_arc_info for the same node
     * @return The node's out arcs
     */
    [[nodiscard]] std::span<const OutArc> get_out_arcs(const mpi_rank_type mpi_rank, const node_id_type node_id, const ArcInfo out_arc_info) const {
        if (mpi_rank == mpiPP::MPIInfo::get_my_rank().get_rank()) {
            const auto span = std::span<const OutArc>(out_arcs_window.get_pointer() + out_arc_info.prefix_arcs, out_arc_info.number_arcs);

            return span;
        }

        if (!out_arc_cache.contains(mpi_rank, node_id)) {
            auto arcs = std::vector<OutArc>(out_arc_info.number_arcs);
            if (!arcs.empty()) {
                out_arcs_window.get(arcs.data(), out_arc_info.number_arcs, out_arc_info.prefix_arcs, mpiPP::MPIRank{ mpi_rank });
            }

            out_arc_cache.insert(mpi_rank, node_id, std::move(arcs));
        }

        return out_arc_cache.get_value(mpi_rank, node_id);
    }

    /**
     * @brief Discards the arcs of the nodes on the other MPI ranks that get_in_arcs and get_out_arcs
     *		cached, so that the following accesses fetch them again. The graph itself is unchanged, only
     *		the communication a traversal performs is; use it to measure an algorithm on a cold cache,
     *		or to bound the memory the cache occupies between two traversals.
     *		Needs no communication and can therefore be called on a single MPI rank.
     */
    void clear_arc_cache() const {
        in_arc_cache.clear();
        out_arc_cache.clear();
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
     * @brief Returns the in-arc info of all local nodes
     * @return The infos, i.e., <return>[node_id].number_arcs == k indicates that local node <node_id> has k in arcs,
     *		which are stored at <return>[node_id].prefix_arcs in the in-arc RMA window
     */
    [[nodiscard]] std::span<const ArcInfo> get_local_in_arc_info_distribution() const {
        const auto size = utility::safe_cast<std::size_t>(in_arc_info_window.get_local_size());
        return { in_arc_info_window.get_pointer(), size };
    }

    /**
     * @brief Returns the distribution of the weight of local in arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has the weight of k in arcs
     */
    [[nodiscard]] std::span<const weight_type> get_weight_in_arc_distribution() const {
        const auto size = utility::safe_cast<std::size_t>(weight_in_arcs_window.get_local_size());
        return { weight_in_arcs_window.get_pointer(), size };
    }

    /**
     * @brief Returns the number of out arcs that start on the current MPI rank
     * @return The number of local out arcs
     */
    [[nodiscard]] arc_id_type get_number_local_out_arcs() const noexcept {
        return local_number_out_arcs;
    }

    /**
     * @brief Returns the out-arc info of all local nodes
     * @return The infos, i.e., <return>[node_id].number_arcs == k indicates that local node <node_id> has k out arcs,
     *		which are stored at <return>[node_id].prefix_arcs in the out-arc RMA window
     */
    [[nodiscard]] std::span<const ArcInfo> get_local_out_arc_info_distribution() const {
        const auto size = utility::safe_cast<std::size_t>(out_arc_info_window.get_local_size());
        return { out_arc_info_window.get_pointer(), size };
    }

    /**
     * @brief Returns the distribution of the weight of local out arcs
     * @return The distribution, i.e., <return>[node_id] == k indicates that local node <node_id> has the weight of k out arcs
     */
    [[nodiscard]] std::span<const weight_type> get_weight_out_arc_distribution() const {
        const auto size = utility::safe_cast<std::size_t>(weight_out_arcs_window.get_local_size());
        return { weight_out_arcs_window.get_pointer(), size };
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
    mpiPP::RMAWindow<ArcInfo> in_arc_info_window;
    mpiPP::RMAWindow<weight_type> weight_in_arcs_window;

    mpiPP::RMAWindow<OutArc> out_arcs_window;
    mpiPP::RMAWindow<ArcInfo> out_arc_info_window;
    mpiPP::RMAWindow<weight_type> weight_out_arcs_window;

    mutable utility::DataCache<InArc, mpi_rank_type, node_id_type> in_arc_cache{};
    mutable utility::DataCache<OutArc, mpi_rank_type, node_id_type> out_arc_cache{};
};
