/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_rma_accumulate.h"

#include "Types.h"

#include "metrics/local_structure/NetworkMotifs.h"
#include "network_motifs/network_motif_expectations.h"

#include <mpi-wrapper/core/MPIInfo.h>

// Every NetworkMotifAlgorithm reports the same fractions, so this variant is checked against the
// shared census of network_motif_expectations.h, which is tied to the numbering of
// NetworkMotifAlgorithm::Questions by the one-rank values of test_network_motifs.cpp.

namespace {
constexpr auto algorithm = NetworkMotifAlgorithm::RmaAccumulate;
} // namespace

namespace census = network_motif_expectations;

TEST_F(NetworkMotifsRmaAccumulateTest, testStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_one_rank);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_uu_one_rank);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::full_one_rank);
}

// The graphs of the four and seven MPI ranks are checked twice: once flattened onto a single rank,
// where no node is remote and the variant therefore cannot mis-assign a triple, and once distributed,
// which is where the ownership rules of the variants actually run. Both have to report the same
// values, so a triple that is counted twice or by nobody shows up as a difference between them.

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_four_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_uu_four_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::full_four_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardFourRanks) {
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

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardUUFourRanks) {
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

TEST_F(NetworkMotifsRmaAccumulateTest, testFullFourRanks) {
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

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_seven_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::standard_uu_seven_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    const auto motifs = NetworkMotifs::compute_network_triple_motifs(graph, algorithm);

    census::expect_motifs(motifs, census::full_seven_ranks);
}

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardSevenRanks) {
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

TEST_F(NetworkMotifsRmaAccumulateTest, testStandardUUSevenRanks) {
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

TEST_F(NetworkMotifsRmaAccumulateTest, testFullSevenRanks) {
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
