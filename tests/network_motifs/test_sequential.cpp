/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_sequential.h"

#include "metrics/local_structure/NetworkMotifs.h"
#include "network_motifs/network_motif_expectations.h"

#include <mpi-wrapper/core/MPIInfo.h>

// The sequential reference implementation is the yardstick the NetworkMotifAlgorithm variants are
// measured against, so it has to report exactly the census of network_motif_expectations.h that
// they are checked against. It classifies a triple with the same bit array as they do, but reaches
// every triple from its first node instead of splitting the work over the ranks, which is what
// makes it an independent check of the ownership rules of the distributed variants.

namespace census = network_motif_expectations;

TEST_F(NetworkMotifsSequentialTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_one_rank);
}

TEST_F(NetworkMotifsSequentialTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_uu_one_rank);
}

TEST_F(NetworkMotifsSequentialTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::full_one_rank);
}

// The graphs of the four and seven MPI ranks are checked twice: once flattened onto a single rank,
// where the traversal stays local, and once distributed, where the root rank reads every adjacency
// of the other ranks. Both have to report the same values.

TEST_F(NetworkMotifsSequentialTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_four_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_uu_four_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::full_four_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    // The values are the corresponding ones from the one-rank version
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_four_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_uu_four_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::full_four_ranks);
}

// The seven-rank graphs consist of three independent parts: the ranks 0, 1, 4, and 6 form the
// four-rank graph, the ranks 2 and 5 form its two-rank version, and rank 3 holds isolated nodes,
// sinks, and sources. They therefore also cover a rank whose nodes take part in almost no triple.

TEST_F(NetworkMotifsSequentialTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_seven_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::standard_uu_seven_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    census::expect_motifs(motifs, census::full_seven_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_seven_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_uu_seven_ranks);
}

TEST_F(NetworkMotifsSequentialTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs_sequential(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::full_seven_ranks);
}
