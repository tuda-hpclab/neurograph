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

#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <mpi-wrapper/core/MPIInfo.h>

#include <array>
#include <cstdint>

/**
 * @brief RmaVariant with the final reduction of the counts done one-sidedly instead of collectively.
 *
 *		The enumeration is the one of RmaVariant and reads the arcs of the other MPI ranks through the
 *		RMA windows of the graph, but the counts are then still collected with a collective reduction,
 *		which makes every rank wait for the slowest one. This variant sums them into a window on the
 *		root rank with MPI_Accumulate instead, so a rank that is done can deposit its counts and leave
 *		without the others having reached that point.
 *
 *		Whether that is worth anything depends entirely on how unevenly the enumeration is distributed;
 *		the payload itself is fourteen integers, so the reduction is never the bottleneck. The variant
 *		exists to measure that difference, and it is the only place in the library that calls MPI
 *		directly, because mpi-wrapper exposes no accumulate operation.
 */
class RmaAccumulateVariant {
public:
    /**
     * @brief Computes the fraction of each three-node network motif in the graph.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(sum over all nodes of the degree squared), neglecting the reduction.
     * @param graph The distributed graph
     * @exception Throws an Exception if MPI reports an error
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
        const auto total_counts = accumulate_counts_on_root(local_counts);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return {};
        }

        return network_motifs::assemble_motif_fractions(total_counts);
    }

private:
    /**
     * @brief Sums the counts of all MPI ranks into a window on the root rank with MPI_Accumulate.
     *		The window is seeded with the counts of the rank that owns it, so the root rank contributes
     *		its own counts without accumulating into itself.
     *		Must be called on every MPI rank.
     * @param local_counts The motifs the current MPI rank counted
     * @exception Throws an Exception if MPI reports an error
     * @return The counts of the whole graph on the root rank, the local ones on every other MPI rank
     */
    [[nodiscard]] static network_motifs::MotifCounts accumulate_counts_on_root(const network_motifs::MotifCounts& local_counts) {
        // The window memory is seeded with the local counts and receives the ones of the other ranks;
        // the origin buffer of the accumulate has to be a separate one, a window may not expose the
        // memory an RMA operation reads from at the same time
        auto total_counts = local_counts;
        const auto number_counts = static_cast<int>(total_counts.size());

        auto window = MPI_Win{};
        auto error = MPI_Win_create(total_counts.data(), static_cast<MPI_Aint>(sizeof(network_motifs::MotifCounts)), static_cast<int>(sizeof(std::uint64_t)), MPI_INFO_NULL,
                                    MPI_COMM_WORLD, &window);
        utility::Exception::check(error == MPI_SUCCESS, "RmaAccumulateVariant: MPI_Win_create failed");

        error = MPI_Win_fence(0, window);
        utility::Exception::check(error == MPI_SUCCESS, "RmaAccumulateVariant: the first MPI_Win_fence failed");

        if (!mpiPP::MPIInfo::is_root_rank()) {
            error = MPI_Accumulate(local_counts.data(), number_counts, MPI_UINT64_T, 0, 0, number_counts, MPI_UINT64_T, MPI_SUM, window);
            utility::Exception::check(error == MPI_SUCCESS, "RmaAccumulateVariant: MPI_Accumulate failed");
        }

        error = MPI_Win_fence(0, window);
        utility::Exception::check(error == MPI_SUCCESS, "RmaAccumulateVariant: the second MPI_Win_fence failed");

        error = MPI_Win_free(&window);
        utility::Exception::check(error == MPI_SUCCESS, "RmaAccumulateVariant: MPI_Win_free failed");

        return total_counts;
    }
};
