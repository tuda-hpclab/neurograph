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
#include "graph/DistributedGraph.h"
#include "metrics/local_structure/NetworkMotifsVariants/MotifClassification.h"

#include <cpp-utility/hash/hash.hpp>
#include <cpp-utility/hash/pair.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace network_motifs {

/**
 * @brief Answers the question whether two nodes of a triple are connected, which is the only question
 *		the RMA variants have to ask about nodes that the current MPI rank does not own.
 *
 *		A single arc cannot be fetched on its own, so answering the question for one pair downloads the
 *		whole adjacency of one of its two nodes. The cache therefore stores the arcs of every pair that
 *		such a download revealed, and remembers which nodes it has downloaded: once the adjacency of a
 *		node is known in full, every pair that contains it is answered from the cache, and a pair that
 *		is not stored is known to be unconnected. That turns the quadratically many pair questions of
 *		one node into at most one download per adjacent node.
 *
 *		The arcs of a pair are stored under the pair ordered by MotifNode, so that both orientations of
 *		a question share one entry; get_arc flips the result back for the caller.
 */
class MotifArcCache {
public:
    /**
     * @brief How the cache picks the node whose adjacency it downloads when the current MPI rank owns
     *		neither node of a pair and both of them are still unknown.
     */
    enum class DownloadChoice : std::uint8_t {
        /**
         * @brief Downloads the adjacency of the first node of the pair, i.e., of the smaller one.
         *		Needs no additional remote memory access to make the decision.
         */
        FirstNode,

        /**
         * @brief Downloads the adjacency of whichever node has fewer arcs, which transfers fewer bytes
         *		but pays four additional remote memory accesses for the arc infos of the two candidates.
         *		Worthwhile when the degrees of the graph differ a lot.
         */
        SmallerAdjacency,
    };

    /**
     * @brief Creates an empty cache for the graph.
     * @param distributed_graph The distributed graph, must outlive the cache
     * @param selected_download_choice How to pick the node whose adjacency is downloaded, see DownloadChoice
     */
    MotifArcCache(const DistributedGraph& distributed_graph, const DownloadChoice selected_download_choice)
        : graph(distributed_graph)
        , my_rank(mpiPP::MPIInfo::get_my_rank().get_rank())
        , download_choice(selected_download_choice) { }

    /**
     * @brief Returns the arcs between the two nodes, downloading the adjacency of one of them if it is
     *		not known yet. Might perform communication via MPI.
     * @param first_node The first node of the pair
     * @param second_node The second node of the pair, must differ from the first one
     * @return The arcs of the pair, oriented as (first_node, second_node)
     */
    [[nodiscard]] ArcType get_arc(const MotifNode first_node, const MotifNode second_node) {
        // One entry per unordered pair, so ask for the ordered one and flip the answer back
        if (second_node < first_node) {
            return flip_arc(get_arc(second_node, first_node));
        }

        if (const auto entry = arcs_of_pair.find({ first_node, second_node }); entry != arcs_of_pair.end()) {
            return entry->second;
        }

        // The adjacency of one of the two nodes is known in full, so a missing entry means no arc
        if (downloaded_nodes.contains(first_node) || downloaded_nodes.contains(second_node)) {
            return no_arc;
        }

        download_adjacency_of_one_node(first_node, second_node);

        if (const auto entry = arcs_of_pair.find({ first_node, second_node }); entry != arcs_of_pair.end()) {
            return entry->second;
        }

        // Remember the negative answer as well, it is settled for good now
        arcs_of_pair[{ first_node, second_node }] = no_arc;
        return no_arc;
    }

private:
    /**
     * @brief Downloads the adjacency of one of the two nodes, preferring a local one and otherwise
     *		following the DownloadChoice of the cache.
     * @param first_node The first node of the pair
     * @param second_node The second node of the pair
     */
    void download_adjacency_of_one_node(const MotifNode first_node, const MotifNode second_node) {
        // A local adjacency needs no communication, so never look at the other node in that case
        if (first_node.first == my_rank) {
            download_adjacency(first_node);
            return;
        }

        if (second_node.first == my_rank) {
            download_adjacency(second_node);
            return;
        }

        if (download_choice == DownloadChoice::FirstNode) {
            download_adjacency(first_node);
            return;
        }

        const auto in_arc_info_first = graph.get_in_arc_info(first_node.first, first_node.second);
        const auto out_arc_info_first = graph.get_out_arc_info(first_node.first, first_node.second);
        const auto in_arc_info_second = graph.get_in_arc_info(second_node.first, second_node.second);
        const auto out_arc_info_second = graph.get_out_arc_info(second_node.first, second_node.second);

        const auto degree_first = in_arc_info_first.number_arcs + out_arc_info_first.number_arcs;
        const auto degree_second = in_arc_info_second.number_arcs + out_arc_info_second.number_arcs;

        if (degree_first < degree_second) {
            download_adjacency(first_node, in_arc_info_first, out_arc_info_first);
        } else {
            download_adjacency(second_node, in_arc_info_second, out_arc_info_second);
        }
    }

    /**
     * @brief Downloads the whole adjacency of the node and stores the arcs of every pair it contains.
     * @param node The node
     */
    void download_adjacency(const MotifNode node) {
        store_adjacency(node, graph.get_in_arcs(node.first, node.second), graph.get_out_arcs(node.first, node.second));
    }

    /**
     * @brief Downloads the whole adjacency of the node, reusing arc infos that were fetched before.
     * @param node The node
     * @param in_arc_info The in-arc info of the node
     * @param out_arc_info The out-arc info of the node
     */
    void download_adjacency(const MotifNode node, const ArcInfo in_arc_info, const ArcInfo out_arc_info) {
        store_adjacency(node, graph.get_in_arcs(node.first, node.second, in_arc_info), graph.get_out_arcs(node.first, node.second, out_arc_info));
    }

    /**
     * @brief Stores the arcs of every pair that the adjacency of the node contains and marks it as
     *		downloaded, i.e., as settled for every pair that contains it.
     * @param node The node
     * @param in_arcs The in arcs of the node
     * @param out_arcs The out arcs of the node
     */
    void store_adjacency(const MotifNode node, const std::span<const InArc> in_arcs, const std::span<const OutArc> out_arcs) {
        downloaded_nodes.emplace(node);

        for (const auto& [target_rank, target_id, _] : out_arcs) {
            store_arc(node, MotifNode{ target_rank, target_id }, ArcType{ 1 });
        }

        for (const auto& [source_rank, source_id, _] : in_arcs) {
            store_arc(node, MotifNode{ source_rank, source_id }, ArcType{ 2 });
        }
    }

    /**
     * @brief Stores one arc of the node under the ordered pair, flipping it if the other node is the
     *		smaller one. Self arcs are dropped, they belong to no node pair.
     * @param node The node whose adjacency was downloaded
     * @param other_node The other end of the arc
     * @param arc The arc, oriented as (node, other_node)
     */
    void store_arc(const MotifNode node, const MotifNode other_node, const ArcType arc) {
        if (node == other_node) {
            return;
        }

        if (node < other_node) {
            merge_arc(arcs_of_pair[{ node, other_node }], arc);
        } else {
            merge_arc(arcs_of_pair[{ other_node, node }], flip_arc(arc));
        }
    }

    const DistributedGraph& graph;
    mpi_rank_type my_rank;
    DownloadChoice download_choice;

    std::unordered_map<std::pair<MotifNode, MotifNode>, ArcType, utility::hash<std::pair<MotifNode, MotifNode>>> arcs_of_pair{};
    std::unordered_set<MotifNode, utility::hash<MotifNode>> downloaded_nodes{};
};

} // namespace network_motifs
