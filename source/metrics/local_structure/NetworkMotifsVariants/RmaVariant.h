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
 * @brief Three-node network motifs by reading the arcs of the other MPI ranks directly, i.e., the base
 *		variant of the RMA family.
 *
 *		Every MPI rank walks its own nodes, takes each of them as the inner node of a triple, and pairs
 *		up its adjacent nodes. Whether the two outer nodes of a pair are connected is the only thing it
 *		cannot answer on its own, and instead of asking their owner a question, as
 *		NetworkMotifs::compute_network_triple_motifs does, it fetches the adjacency it needs through the
 *		RMA windows of the graph (see MotifArcCache). That removes the collective question-and-answer
 *		rounds from the algorithm, so the ranks never wait for each other until the final reduction, but
 *		it also means that a rank may download the adjacency of a node that another rank has in memory.
 *
 *		The classification is a single lookup in motif_of_arc_structure, since a rank that holds all
 *		three pairs of arcs knows the motif of the triple outright.
 */
class RmaVariant {
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
            .remote_pair_handling = network_motifs::RemotePairHandling::LookUpEveryPair,
            .download_choice = network_motifs::MotifArcCache::DownloadChoice::FirstNode,
        };

        const auto local_counts = network_motifs::count_motifs_of_my_rank(graph, strategy);

        return network_motifs::reduce_and_assemble(local_counts);
    }
};
