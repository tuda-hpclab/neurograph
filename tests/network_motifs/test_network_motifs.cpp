/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_network_motifs.h"

#include "Types.h"

#include "metrics/local_structure/NetworkMotifs.h"
#include "network_motifs/network_motif_expectations.h"

#include <mpi-wrapper/core/MPIInfo.h>

// NetworkMotifAlgorithm::Questions is the reference variant, so the expected values of
// network_motif_expectations.h are the ones of this file; the other variants are checked against
// the same census in their own test files.

namespace {
constexpr auto algorithm = NetworkMotifAlgorithm::Questions;
} // namespace

namespace census = network_motif_expectations;

TEST_F(NetworkMotifsTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    // result[0] is the number of connected triples of the graph, result[i] the fraction of motif i
    census::expect_motifs(motifs, census::standard_one_rank);
}

TEST_F(NetworkMotifsTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    // Symmetrized graph: every arc is bidirectional, so only the mutual motifs 9 (open triad) and
    // 13 (mutual triangle) occur. The 11 triangles were confirmed independently via wedge counting
    // (sum of C(deg, 2) = 71 = 3 * triangles + open triads).
    census::expect_motifs(motifs, census::standard_uu_one_rank);
}

TEST_F(NetworkMotifsTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    // Complete graph on 10 nodes: every one of the C(10, 3) = 120 triples is a mutual triangle
    census::expect_motifs(motifs, census::full_one_rank);
}

// The graphs of the four and seven MPI ranks are checked twice: once flattened onto a single rank,
// where every node of a triple is local, and once distributed, where the arcs between two nodes of
// a triple belong to another rank and have to be asked for. Both have to report the same values.

TEST_F(NetworkMotifsTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_four_ranks);
}

TEST_F(NetworkMotifsTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_uu_four_ranks);
}

TEST_F(NetworkMotifsTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::full_four_ranks);
}

TEST_F(NetworkMotifsTest, testStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_four_ranks);
}

TEST_F(NetworkMotifsTest, testStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_uu_four_ranks);
}

TEST_F(NetworkMotifsTest, testFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::full_four_ranks);
}

// The seven-rank graphs consist of three independent parts: the ranks 0, 1, 4, and 6 form the
// four-rank graph, the ranks 2 and 5 form its two-rank version, and rank 3 holds isolated nodes,
// sinks, and sources. They therefore also cover a rank whose nodes take part in almost no triple.

TEST_F(NetworkMotifsTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_seven_ranks);
}

TEST_F(NetworkMotifsTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_uu_seven_ranks);
}

TEST_F(NetworkMotifsTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::full_seven_ranks);
}

TEST_F(NetworkMotifsTest, testStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    // The values are the corresponding ones from the one-rank version
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_seven_ranks);
}

TEST_F(NetworkMotifsTest, testStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::standard_uu_seven_ranks);
}

TEST_F(NetworkMotifsTest, testFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    census::expect_motifs(motifs, census::full_seven_ranks);
}
