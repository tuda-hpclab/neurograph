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

#include "graph/DistributedGraph.h"
#include "metrics/local_structure/NetworkMotifsVariants/MotifArcCache.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaCommon.h"

#include <array>

/**
 * @brief RmaVariant with the triangles that span three MPI ranks spread evenly over their three owners.
 *
 *		RmaVariant gives every such triangle to the owner of its smallest node, which is cheap to decide
 *		but systematically unfair: all triangles of a node go to the same MPI rank, so the ranks that own
 *		the small nodes carry the whole load while the others only answer arc reads. This variant instead
 *		lets the three owners take turns, selected by the local id of the smallest node of the triple
 *		(see network_motifs::is_my_turn_for_triangle), which splits the triangles of one rank into three
 *		roughly equal parts without any communication about the decision.
 *
 *		It is the better choice whenever the triangles of the graph reach across many MPI ranks; on a
 *		graph whose triangles are local, the turn computation is pure overhead, because a triangle with
 *		two nodes on one rank is settled by the same rule as in RmaVariant.
 */
class RmaFairVariant {
public:
    /**
     * @brief Computes the fraction of each three-node network motif in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(sum over all nodes of the degree squared), neglecting the reduction.
     * @param graph The distributed graph
     * @return The motif fractions in the layout of NetworkMotifs::compute_network_triple_motifs;
     *		meaningful only on MPI rank 0
     */
    [[nodiscard]] static std::array<long double, 14> all_compute_motifs(const DistributedGraph& graph) {
        constexpr auto strategy = network_motifs::RmaStrategy{
            .triangle_owner = network_motifs::TriangleOwner::RoundRobin,
            .remote_pair_handling = network_motifs::RemotePairHandling::LookUpEveryPair,
            .download_choice = network_motifs::MotifArcCache::DownloadChoice::FirstNode,
        };

        const auto local_counts = network_motifs::count_motifs_of_my_rank(graph, strategy);

        return network_motifs::reduce_and_assemble(local_counts);
    }
};
