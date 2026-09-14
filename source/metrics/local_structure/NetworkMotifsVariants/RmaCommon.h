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
#include "metrics/local_structure/NetworkMotifsVariants/MotifArcCache.h"
#include "metrics/local_structure/NetworkMotifsVariants/MotifClassification.h"

#include <cpp-utility/hash/hash.hpp>
#include <cpp-utility/hash/pair.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace network_motifs {

/**
 * @brief How many motifs of each type the current MPI rank counted; index 0 stays unused so that the
 *		indices are the motif numbers.
 */
using MotifCounts = std::array<std::uint64_t, 14>;

/**
 * @brief Which MPI rank counts a triple whose three nodes lie on three different MPI ranks and that is
 *		closed, i.e., a triangle. All three owners see such a triple and would count it three times,
 *		so exactly one of them has to.
 */
enum class TriangleOwner : std::uint8_t {
    /**
     * @brief The owner of the smallest node of the triple counts it. Needs no computation, but gives
     *		all triples of a node to the same MPI rank, which is why the ranks that own small nodes do
     *		more work than the others.
     */
    SmallestNode,

    /**
     * @brief The three owners take turns, selected by the local id of the smallest node of the triple.
     *		Spreads the triples of one node over all three owners, at the price of computing the turn.
     */
    RoundRobin,
};

/**
 * @brief Whether a variant defers the triples whose two outer nodes lie on one single other MPI rank,
 *		which lets it skip the arc lookups for them, see enumerate_local_motifs.
 */
enum class RemotePairHandling : std::uint8_t {
    /**
     * @brief Looks every pair up, i.e., asks the arc cache about the two outer nodes of every triple
     */
    LookUpEveryPair,

    /**
     * @brief Skips a triple whose two outer nodes lie on one single other MPI rank, because that rank
     *		sees the same triple with the two nodes it owns as the inner and one outer node, and counts
     *		the closed ones itself. The open ones are not centered on one of its nodes, so they are
     *		collected separately, see enumerate_deferred_motifs.
     */
    DeferPairsOnOneRemoteRank,
};

/**
 * @brief The strategy of one RMA variant, i.e., the three decisions in which the variants differ while
 *		they enumerate the triples. How the counts are reduced afterwards is up to the variant itself.
 */
struct RmaStrategy {
    /**
     * @brief Which MPI rank counts a triangle that spans three MPI ranks
     */
    TriangleOwner triangle_owner{ TriangleOwner::SmallestNode };

    /**
     * @brief Whether triples with both outer nodes on one other MPI rank are deferred
     */
    RemotePairHandling remote_pair_handling{ RemotePairHandling::LookUpEveryPair };

    /**
     * @brief Which node of a pair the arc cache downloads, see MotifArcCache::DownloadChoice
     */
    MotifArcCache::DownloadChoice download_choice{ MotifArcCache::DownloadChoice::FirstNode };
};

/**
 * @brief One node adjacent to another one, together with the arcs that connect the two
 */
struct AdjacentNode {
    /**
     * @brief The adjacent node
     */
    MotifNode node{};

    /**
     * @brief The arcs between the two nodes, oriented as (the node the adjacency belongs to, this node)
     */
    ArcType arcs{};
};

/**
 * @brief The local nodes that are adjacent to one node on another MPI rank, collected while the
 *		triples are enumerated, see enumerate_deferred_motifs.
 */
using RemoteNeighborhoods = std::unordered_map<MotifNode, std::unordered_set<MotifNode, utility::hash<MotifNode>>, utility::hash<MotifNode>>;

/**
 * @brief Decides whether the current MPI rank is the one that counts a triangle that spans three MPI
 *		ranks, so that exactly one of the three owners does.
 *
 *		All three owners enumerate the triangle once, each of them with the node it owns as node 1, so
 *		the decision has to depend on the triple only and not on which of the three nodes is node 1.
 *		Sorting the triple gives the same order to all three, and the local id of its smallest node
 *		then selects one of the three positions, which is why every third node of a rank sends its
 *		triangles to a different owner.
 * @param node_1 The node of the triple that the current MPI rank owns
 * @param node_2 The second node of the triple
 * @param node_3 The third node of the triple
 * @return True iff the current MPI rank has to count the triangle
 */
[[nodiscard]] inline bool is_my_turn_for_triangle(const MotifNode node_1, const MotifNode node_2, const MotifNode node_3) {
    // The smallest node of the triple selects the position that counts it: 0 the smallest node,
    // 1 the middle one, and 2 the largest one
    if (node_1 < node_2 && node_1 < node_3) {
        return node_1.second % 3U == 0;
    }

    const auto smallest_node = node_2 < node_3 ? node_2 : node_3;
    const auto largest_other_node = node_2 < node_3 ? node_3 : node_2;

    if (node_1 < largest_other_node) {
        return smallest_node.second % 3U == 1;
    }

    return smallest_node.second % 3U == 2;
}

/**
 * @brief Collects the nodes adjacent to a local node, i.e., the candidates for the two outer nodes of
 *		its triples. A node that is both an in and an out neighbor appears once, with both arcs set, and
 *		self arcs are dropped, as they belong to no triple.
 *
 *		The result is a vector and not a map because the enumeration pairs its entries up, which is
 *		quadratic in the degree and therefore the hot loop of every variant.
 * @param graph The distributed graph
 * @param node The local node
 * @return Its adjacent nodes with the arcs that connect them to it, in no particular order
 */
[[nodiscard]] inline std::vector<AdjacentNode> collect_adjacent_nodes(const DistributedGraph& graph, const MotifNode node) {
    const auto in_arcs = graph.get_in_arcs(node.first, node.second);
    const auto out_arcs = graph.get_out_arcs(node.first, node.second);

    auto index_of_node = std::unordered_map<MotifNode, std::size_t, utility::hash<MotifNode>>{};
    index_of_node.reserve(in_arcs.size() + out_arcs.size());

    auto adjacent_nodes = std::vector<AdjacentNode>{};
    adjacent_nodes.reserve(in_arcs.size() + out_arcs.size());

    const auto add_arc = [&index_of_node, &adjacent_nodes, node](const MotifNode other_node, const ArcType arc) {
        if (other_node == node) {
            return;
        }

        const auto [entry, inserted] = index_of_node.emplace(other_node, adjacent_nodes.size());
        if (inserted) {
            adjacent_nodes.push_back(AdjacentNode{ .node = other_node, .arcs = arc });
        } else {
            merge_arc(adjacent_nodes[entry->second].arcs, arc);
        }
    };

    for (const auto& [target_rank, target_id, _] : out_arcs) {
        add_arc(MotifNode{ target_rank, target_id }, ArcType{ 1 });
    }

    for (const auto& [source_rank, source_id, _] : in_arcs) {
        add_arc(MotifNode{ source_rank, source_id }, ArcType{ 2 });
    }

    return adjacent_nodes;
}

/**
 * @brief Decides whether the current MPI rank counts a triangle it enumerated, so that exactly one of
 *		the MPI ranks that see it does.
 *
 *		All three nodes of a triangle see it, each of them as the inner node of the triple. Which of
 *		them keeps it depends on how the nodes are distributed and on the strategy:
 *		  - if the three nodes lie on three different MPI ranks, all three owners enumerate it once, and
 *		    either the owner of the smallest node keeps it or the three take turns, see TriangleOwner,
 *		  - otherwise at least two of the nodes lie on one MPI rank, which therefore enumerates the
 *		    triangle more than once, and the view whose inner node is the smallest one keeps it.
 *
 *		A variant that defers the pairs on one other MPI rank needs a weaker rule in the second case:
 *		it never enumerates a triangle whose two outer nodes lie together on another MPI rank, so if a
 *		remote outer node could veto a view, the triangle would end up counted by nobody. Only a
 *		smaller outer node on the current MPI rank may veto there, and that suffices, because the
 *		remaining views all belong to the rank that owns at least two of the nodes.
 * @param my_rank The current MPI rank
 * @param node_1 The inner node of the triple, owned by the current MPI rank
 * @param node_2 The first outer node
 * @param node_3 The second outer node
 * @param strategy The strategy of the variant
 * @return True iff the current MPI rank has to count the triangle
 */
[[nodiscard]] inline bool owns_triangle(const mpi_rank_type my_rank, const MotifNode node_1, const MotifNode node_2, const MotifNode node_3,
                                        const RmaStrategy strategy) {
    const auto spans_three_ranks = node_2.first != node_3.first && node_2.first != my_rank && node_3.first != my_rank;

    if (spans_three_ranks) {
        if (strategy.triangle_owner == TriangleOwner::RoundRobin) {
            return is_my_turn_for_triangle(node_1, node_2, node_3);
        }

        return node_1 < node_2 && node_1 < node_3;
    }

    if (strategy.remote_pair_handling == RemotePairHandling::DeferPairsOnOneRemoteRank) {
        const auto vetoed_by_node_2 = node_2 < node_1 && node_2.first == my_rank;
        const auto vetoed_by_node_3 = node_3 < node_1 && node_3.first == my_rank;

        return !vetoed_by_node_2 && !vetoed_by_node_3;
    }

    return node_1 < node_2 && node_1 < node_3;
}

/**
 * @brief Adds one classified triple to the counts.
 * @param counts The counts, updated in place
 * @param arc_structure The arc structure of the triple, must be one the enumeration can build
 */
inline void count_arc_structure(MotifCounts& counts, const ArcStructure arc_structure) {
    const auto motif = motif_of_arc_structure[arc_structure];
    assert(motif != unreachable_motif && "network_motifs: the enumeration built an arc structure without a motif");

    counts[static_cast<std::size_t>(motif)]++;
}

/**
 * @brief Enumerates every triple that is centered on a node of the current MPI rank and counts the
 *		motifs the strategy assigns to it.
 *
 *		A triple consists of an inner node, which the current MPI rank owns, and two outer nodes
 *		adjacent to it. Every open triple, i.e., every one whose outer nodes are not connected, has
 *		exactly one inner node and is therefore enumerated exactly once. A closed triple is a triangle
 *		and is enumerated by all three of its nodes, so owns_triangle decides which of those views
 *		counts it.
 * @param graph The distributed graph
 * @param arc_cache The arc cache that answers whether the two outer nodes are connected
 * @param strategy The strategy of the variant
 * @param remote_neighborhoods Receives the local nodes adjacent to each node on another MPI rank; only
 *		filled if the strategy defers the pairs on one other MPI rank
 * @return The motifs the current MPI rank counted
 */
[[nodiscard]] inline MotifCounts enumerate_local_motifs(const DistributedGraph& graph, MotifArcCache& arc_cache, const RmaStrategy strategy,
                                                        RemoteNeighborhoods& remote_neighborhoods) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = graph.get_number_local_nodes();
    const auto defers_remote_pairs = strategy.remote_pair_handling == RemotePairHandling::DeferPairsOnOneRemoteRank;

    auto counts = MotifCounts{};

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto node_1 = MotifNode{ my_rank, node_id };
        const auto adjacent_nodes = collect_adjacent_nodes(graph, node_1);

        // Pair every two adjacent nodes up exactly once, which the index bounds guarantee; which of
        // the two becomes node 2 and which node 3 does not matter, swapping them mirrors the triple
        // and a mirrored triple has the same motif
        for (auto outer = std::size_t{ 0 }; outer < adjacent_nodes.size(); ++outer) {
            const auto [node_2, arcs_node_1_2] = adjacent_nodes[outer];

            if (defers_remote_pairs && node_2.first != my_rank) {
                remote_neighborhoods[node_2].emplace(node_1);
            }

            for (auto inner = std::size_t{ 0 }; inner < outer; ++inner) {
                const auto [node_3, arcs_node_1_3] = adjacent_nodes[inner];

                // The rank that owns both outer nodes sees this triple as well, and counts it if it
                // is closed; the open ones are collected by enumerate_deferred_motifs
                if (defers_remote_pairs && node_2.first != my_rank && node_2.first == node_3.first) {
                    continue;
                }

                const auto arcs_node_2_3 = arc_cache.get_arc(node_2, node_3);

                if (arcs_node_2_3 != no_arc && !owns_triangle(my_rank, node_1, node_2, node_3, strategy)) {
                    continue;
                }

                count_arc_structure(counts, build_arc_structure(arcs_node_1_2, arcs_node_2_3, arcs_node_1_3));
            }
        }
    }

    return counts;
}

/**
 * @brief Counts the open triples that the deferral of enumerate_local_motifs left out, i.e., the ones
 *		whose inner node lies on another MPI rank and whose two outer nodes lie on the current one.
 *
 *		Such a triple is centered on a remote node, so no rank sees it while it walks its own nodes.
 *		The rank that owns both outer nodes is the natural one to count it, and it knows every remote
 *		node its own nodes are adjacent to from remote_neighborhoods. Only the open triples are counted
 *		here; a closed one is a triangle, which the owner of the inner node has already counted.
 * @param arc_cache The arc cache that answers whether two nodes are connected
 * @param remote_neighborhoods The local nodes adjacent to each node on another MPI rank
 * @return The motifs the current MPI rank counted in addition
 */
[[nodiscard]] inline MotifCounts enumerate_deferred_motifs(MotifArcCache& arc_cache, const RemoteNeighborhoods& remote_neighborhoods) {
    auto counts = MotifCounts{};

    for (const auto& [remote_node, local_neighbors] : remote_neighborhoods) {
        // Advancing the inner iterator past the outer one pairs every two neighbors up exactly once
        for (auto outer = local_neighbors.begin(); outer != local_neighbors.end(); ++outer) {
            for (auto inner = std::next(outer); inner != local_neighbors.end(); ++inner) {
                const auto node_2 = *outer;
                const auto node_3 = *inner;

                // A closed triple is a triangle, which the owner of the remote node counted already
                if (arc_cache.get_arc(node_2, node_3) != no_arc) {
                    continue;
                }

                const auto arcs_node_1_2 = arc_cache.get_arc(remote_node, node_2);
                const auto arcs_node_1_3 = arc_cache.get_arc(remote_node, node_3);

                count_arc_structure(counts, build_arc_structure(arcs_node_1_2, no_arc, arcs_node_1_3));
            }
        }
    }

    return counts;
}

/**
 * @brief Counts the motifs the current MPI rank is responsible for, i.e., runs the enumeration of the
 *		strategy including the deferred triples if it uses them. Performs no collective communication,
 *		but reads the arcs of the other MPI ranks via RMA.
 * @param graph The distributed graph
 * @param strategy The strategy of the variant
 * @return The motifs the current MPI rank counted
 */
[[nodiscard]] inline MotifCounts count_motifs_of_my_rank(const DistributedGraph& graph, const RmaStrategy strategy) {
    auto arc_cache = MotifArcCache{ graph, strategy.download_choice };
    auto remote_neighborhoods = RemoteNeighborhoods{};

    auto counts = enumerate_local_motifs(graph, arc_cache, strategy, remote_neighborhoods);

    if (strategy.remote_pair_handling == RemotePairHandling::DeferPairsOnOneRemoteRank) {
        const auto deferred_counts = enumerate_deferred_motifs(arc_cache, remote_neighborhoods);

        for (auto motif_type = std::size_t{ 0 }; motif_type < counts.size(); ++motif_type) {
            counts[motif_type] += deferred_counts[motif_type];
        }
    }

    return counts;
}

/**
 * @brief Turns the motif counts of the whole graph into the result of NetworkMotifs, i.e., the total
 *		number of motifs and the fraction of each type.
 * @param counts The counts of the whole graph
 * @return The total count in index 0 and the fractions in the indices 1 to 13
 */
[[nodiscard]] inline std::array<long double, 14> assemble_motif_fractions(const MotifCounts& counts) {
    const auto total_number_motifs = std::accumulate(counts.begin(), counts.end(), std::uint64_t{ 0 });

    auto motif_fractions = std::array<long double, 14>{};
    motif_fractions[0] = static_cast<long double>(total_number_motifs);

    for (auto motif_type = std::size_t{ 1 }; motif_type < motif_fractions.size(); ++motif_type) {
        motif_fractions[motif_type] = static_cast<long double>(counts[motif_type]) / motif_fractions[0];
    }

    return motif_fractions;
}

/**
 * @brief Sums the motif counts of all MPI ranks on the root rank and assembles the result there.
 *		Must be called on every MPI rank; only the root rank receives the result, as the other
 *		variants of NetworkMotifs do it as well.
 * @param local_counts The motifs the current MPI rank counted
 * @return The result on the root rank, an all-zero array on every other MPI rank
 */
[[nodiscard]] inline std::array<long double, 14> reduce_and_assemble(const MotifCounts& local_counts) {
    const auto total_counts = mpiPP::MPIReductions::reduce_componentwise_sum(local_counts);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return {};
    }

    return assemble_motif_fractions(total_counts);
}

} // namespace network_motifs
