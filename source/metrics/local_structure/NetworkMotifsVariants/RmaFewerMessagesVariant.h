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
 * @brief RmaVariant that leaves the triples whose two outer nodes sit together on one other MPI rank
 *		to that rank, so that it never has to read arcs for them.
 *
 *		Those triples are the expensive ones for RmaVariant: both outer nodes are remote, so deciding
 *		whether they are connected always costs a download, while the rank that owns them both knows
 *		the answer without any communication. This variant therefore skips them and lets the other rank
 *		do the work, which it can, because it sees the same node triple with one of its own nodes as
 *		the inner one.
 *
 *		That covers the closed triples. An open one has only one inner node, so if that node is remote
 *		while both outer nodes are local, no rank encounters it while walking its own nodes. The variant
 *		collects those separately: while it enumerates, it records for every remote node which of its
 *		own nodes are adjacent to it, and afterwards pairs those up (see
 *		network_motifs::enumerate_deferred_motifs). The pairs it needs are all local, so the second pass
 *		adds no communication of its own.
 *
 *		The result is less RMA traffic at the price of a second pass and the bookkeeping of the remote
 *		neighborhoods, which grows with the number of arcs that cross MPI ranks.
 */
class RmaFewerMessagesVariant {
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
            .triangle_owner = network_motifs::TriangleOwner::SmallestNode,
            .remote_pair_handling = network_motifs::RemotePairHandling::DeferPairsOnOneRemoteRank,
            .download_choice = network_motifs::MotifArcCache::DownloadChoice::FirstNode,
        };

        const auto local_counts = network_motifs::count_motifs_of_my_rank(graph, strategy);

        return network_motifs::reduce_and_assemble(local_counts);
    }
};
