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
 * @brief RmaVariant that downloads the adjacency of whichever outer node has fewer arcs.
 *
 *		Whether two outer nodes are connected can be read off the adjacency of either of them, and
 *		RmaVariant simply takes the first one. This variant compares the degrees of the two candidates
 *		first and downloads the smaller adjacency, which trades four remote memory accesses for the arc
 *		infos against the arcs it then does not transfer.
 *
 *		That pays off on graphs with a skewed degree distribution, where the arcs of a hub dwarf the
 *		four extra accesses, and costs latency on a graph whose nodes all have a similar degree. Note
 *		that only the choice differs: the variant counts exactly the same triples as RmaVariant, since
 *		both adjacencies answer the same question.
 */
class RmaFewerArcsVariant {
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
            .download_choice = network_motifs::MotifArcCache::DownloadChoice::SmallerAdjacency,
        };

        const auto local_counts = network_motifs::count_motifs_of_my_rank(graph, strategy);

        return network_motifs::reduce_and_assemble(local_counts);
    }
};
