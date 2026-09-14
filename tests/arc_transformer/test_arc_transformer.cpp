/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc_transformer.h"

#include "Types.h"

#include "graph/ArcTransformer.h"
#include "graph/GraphTypes.h"

#include <gtest/gtest.h>

#include <utility>

namespace {
using arc_collection = LoadedArcs::value_type;

[[nodiscard]] std::pair<std::pair<mpi_rank_type, node_id_type>, weight_type> ce(const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
    return std::make_pair(std::make_pair(rank, node), weight);
}

[[nodiscard]] std::pair<mpi_rank_type, node_id_type> key(const mpi_rank_type rank, const node_id_type node) {
    return std::make_pair(rank, node);
}
} // namespace

TEST_F(ArcTransformerTest, removeWeightZeroRemovesOnlyZeroWeightArcs) {
    auto arcs = LoadedArcs{};
    arcs.emplace_back(arc_collection{ ce(0, 1, 0), ce(0, 2, 5), ce(0, 3, 0) });
    arcs.emplace_back(arc_collection{ ce(0, 0, 0) });
    arcs.emplace_back(arc_collection{ ce(0, 0, 7), ce(0, 1, 2) });

    const auto result = ArcTransformer::remove_weight_zero(arcs);

    ASSERT_EQ(result.size(), 3);

    // node 0: only the weight-5 arc survives
    ASSERT_EQ(result[0].size(), 1);
    ASSERT_TRUE(result[0].contains(key(0, 2)));
    ASSERT_EQ(result[0].at(key(0, 2)), 5);

    // node 1: the single weight-0 arc is removed
    ASSERT_TRUE(result[1].empty());

    // node 2: both non-zero arcs survive
    ASSERT_EQ(result[2].size(), 2);
    ASSERT_EQ(result[2].at(key(0, 0)), 7);
    ASSERT_EQ(result[2].at(key(0, 1)), 2);
}

TEST_F(ArcTransformerTest, removeSelfArcsRemovesArcsToSameNodeOnThisRank) {
    const auto my_rank = mpi_rank_type{ 0 };

    auto arcs = LoadedArcs{};
    arcs.emplace_back(arc_collection{ ce(0, 0, 3), ce(0, 1, 4) }); // node 0 has a self arc (0, 0)
    arcs.emplace_back(arc_collection{ ce(1, 1, 2), ce(0, 1, 5) }); // node 1: (0, 1) is a self arc, (1, 1) is on another rank

    const auto result = ArcTransformer::remove_self_arcs(arcs, my_rank);

    ASSERT_FALSE(result[0].contains(key(0, 0))); // removed
    ASSERT_TRUE(result[0].contains(key(0, 1)));  // kept

    ASSERT_FALSE(result[1].contains(key(0, 1))); // removed (same rank, same node id)
    ASSERT_TRUE(result[1].contains(key(1, 1)));  // kept (different rank)
}

TEST_F(ArcTransformerTest, allWeightsOneSetsEveryWeightToOne) {
    auto arcs = LoadedArcs{};
    arcs.emplace_back(arc_collection{ ce(0, 1, 5), ce(0, 2, 9) });
    arcs.emplace_back(arc_collection{ ce(0, 0, -3) });

    const auto result = ArcTransformer::all_weights_one(arcs);

    ASSERT_EQ(result[0].at(key(0, 1)), 1);
    ASSERT_EQ(result[0].at(key(0, 2)), 1);
    ASSERT_EQ(result[1].at(key(0, 0)), 1);
}

TEST_F(ArcTransformerTest, allWeightsPositiveTakesAbsoluteValue) {
    auto arcs = LoadedArcs{};
    arcs.emplace_back(arc_collection{ ce(0, 1, -5), ce(0, 2, 9) });

    const auto result = ArcTransformer::all_weights_positive(arcs);

    ASSERT_EQ(result[0].at(key(0, 1)), 5);
    ASSERT_EQ(result[0].at(key(0, 2)), 9);
}

TEST_F(ArcTransformerTest, synchronizeArcsMergesIncomingIntoOutgoing) {
    auto incoming = LoadedArcs{};
    incoming.emplace_back(arc_collection{ ce(0, 2, 1) }); // node 0 <- 2
    incoming.emplace_back(arc_collection{ ce(0, 0, 1) }); // node 1 <- 0

    auto outgoing = LoadedArcs{};
    outgoing.emplace_back(arc_collection{ ce(0, 1, 1) }); // node 0 -> 1
    outgoing.emplace_back(arc_collection{ ce(0, 2, 1) }); // node 1 -> 2

    const auto result = ArcTransformer::synchronize_arcs(incoming, outgoing);

    ASSERT_EQ(result.size(), 2);

    // node 0: out {1} union in {2}
    ASSERT_EQ(result[0].size(), 2);
    ASSERT_TRUE(result[0].contains(key(0, 1)));
    ASSERT_TRUE(result[0].contains(key(0, 2)));

    // node 1: out {2} union in {0}
    ASSERT_EQ(result[1].size(), 2);
    ASSERT_TRUE(result[1].contains(key(0, 2)));
    ASSERT_TRUE(result[1].contains(key(0, 0)));
}

TEST_F(ArcTransformerTest, synchronizeArcsClampsCancelledWeightsToOne) {
    auto incoming = LoadedArcs{};
    incoming.emplace_back(arc_collection{ ce(0, 1, -3) });

    auto outgoing = LoadedArcs{};
    outgoing.emplace_back(arc_collection{ ce(0, 1, 3) }); // 3 + (-3) == 0 -> clamped to 1

    const auto result = ArcTransformer::synchronize_arcs(incoming, outgoing);

    ASSERT_EQ(result[0].at(key(0, 1)), 1);
}
