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

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

/**
 * @brief The shared vocabulary of the RMA network-motif variants: how a node is identified, how the
 *		arcs between two nodes are packed into two bits, and how the six bits of a node triple are
 *		mapped onto the motif numbering of NetworkMotifs.
 *
 *		The variants classify a triple in one step from a six-bit arc structure instead of narrowing a
 *		set of candidate motifs like NetworkMotifs::compute_network_triple_motifs does, which is what
 *		lets them work without asking the owner of node 2 a question.
 */
namespace network_motifs {

/**
 * @brief A node of the distributed graph, i.e., its owning MPI rank and its id on that rank.
 *		Ordered lexicographically, which the variants use to give every motif a unique owner.
 */
using MotifNode = std::pair<mpi_rank_type, node_id_type>;

/**
 * @brief The arcs between an ordered pair of nodes (first, second), packed into two bits:
 *		bit 0 is set iff the arc first -> second exists, bit 1 iff the arc second -> first exists.
 */
using ArcType = std::uint8_t;

/**
 * @brief The value of ArcType for a pair of nodes that no arc connects
 */
inline constexpr auto no_arc = ArcType{ 0 };

/**
 * @brief Swaps the two directions of an ArcType, i.e., turns the arcs of the pair (first, second)
 *		into the arcs of the pair (second, first).
 * @param arc The arcs of the pair
 * @return The arcs of the reversed pair
 */
[[nodiscard]] constexpr ArcType flip_arc(const ArcType arc) noexcept {
    const auto forward = static_cast<ArcType>((arc & 1U) << 1U);
    const auto backward = static_cast<ArcType>((arc & 2U) >> 1U);
    return static_cast<ArcType>(forward | backward);
}

/**
 * @brief Adds one direction to the arcs of a node pair, i.e., the |= that ArcType would need a cast
 *		for, because the integral promotion of its operands widens the result.
 * @param arcs The arcs of the pair, updated in place
 * @param direction The direction to add
 */
constexpr void merge_arc(ArcType& arcs, const ArcType direction) noexcept {
    arcs = static_cast<ArcType>(arcs | direction);
}

/**
 * @brief The six bits that describe all arcs of a node triple (node 1, node 2, node 3). The three
 *		pairs are arranged as the cycle 1 -> 2 -> 3 -> 1, so the ArcType of one pair occupies two
 *		neighboring bits and the whole structure is
 *
 *		    bit 0: node 1 -> node 2      bit 1: node 2 -> node 1
 *		    bit 2: node 2 -> node 3      bit 3: node 3 -> node 2
 *		    bit 4: node 3 -> node 1      bit 5: node 1 -> node 3
 *
 *		i.e., an arc structure is assembled as
 *		    arcs(1, 2) | arcs(2, 3) << 2 | flip_arc(arcs(1, 3)) << 4.
 */
using ArcStructure = std::uint8_t;

/**
 * @brief The number of distinct arc structures, i.e., the size of motif_of_arc_structure
 */
inline constexpr auto number_arc_structures = std::size_t{ 64 };

/**
 * @brief Maps an ArcStructure onto the number of its three-node motif, using the numbering of
 *		NetworkMotifs::compute_network_triple_motifs (Kaiser, 2011, page 899). The 64 arc structures
 *		fall into 16 isomorphism classes, 13 of which are connected; this table labels each of them
 *		with its motif number, which makes the classification a single array lookup.
 *
 *		Only the 36 arc structures that the variants can actually build carry a motif number: they only
 *		ever classify a triple in which node 1 is adjacent to both other nodes, so at least one of the
 *		bits 0 and 1 and at least one of the bits 4 and 5 are set. Every other entry is
 *		unreachable_motif, and the variants assert that they never look one up.
 */
inline constexpr auto motif_of_arc_structure = std::array<std::int8_t, number_arc_structures>{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0 - 9
    -1, -1, -1, -1, -1, -1, -1, 2, 1, 4,    // 10 - 19
    -1, 7, 5, 10, -1, 5, 5, 8, -1, 10,      // 20 - 29
    11, 12, -1, 3, 2, 6, -1, 5, 5, 11,      // 30 - 39
    -1, 5, 7, 10, -1, 8, 10, 12, -1, 6,     // 40 - 49
    4, 9, -1, 10, 8, 12, -1, 11, 10, 12,    // 50 - 59
    -1, 12, 12, 13                          // 60 - 63
};

/**
 * @brief The entry of motif_of_arc_structure for an arc structure the variants cannot build
 */
inline constexpr auto unreachable_motif = std::int8_t{ -1 };

/**
 * @brief Assembles the arc structure of a node triple from the arcs of its three pairs.
 * @param arcs_node_1_2 The arcs between node 1 and node 2, oriented as (node 1, node 2)
 * @param arcs_node_2_3 The arcs between node 2 and node 3, oriented as (node 2, node 3)
 * @param arcs_node_1_3 The arcs between node 1 and node 3, oriented as (node 1, node 3)
 * @return The arc structure of the triple
 */
[[nodiscard]] constexpr ArcStructure build_arc_structure(const ArcType arcs_node_1_2, const ArcType arcs_node_2_3, const ArcType arcs_node_1_3) noexcept {
    const auto arcs_node_3_1 = flip_arc(arcs_node_1_3);
    return static_cast<ArcStructure>(arcs_node_1_2 | (arcs_node_2_3 << 2U) | (arcs_node_3_1 << 4U));
}

} // namespace network_motifs
