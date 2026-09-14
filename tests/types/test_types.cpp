/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_types.h"

#include "Types.h"

#include <gtest/gtest.h>

#include <limits>
#include <tuple>
#include <utility>

TEST_F(TypesTest, defaultConstructedNodeIdentifierIsZeroInitialized) {
    NodeIdentifier ni;

    ASSERT_EQ(ni.owning_mpi_rank, mpi_rank_type{ 0 });
    ASSERT_EQ(ni.node_id, node_id_type{ 0 });
    ASSERT_FALSE(ni.has_uninitialized_rank());
}

TEST_F(TypesTest, uninitializedHasMaxRankAndNodeId) {
    const auto ni = NodeIdentifier::uninitialized();

    ASSERT_EQ(ni.owning_mpi_rank, std::numeric_limits<mpi_rank_type>::max());
    ASSERT_EQ(ni.node_id, std::numeric_limits<node_id_type>::max());
    ASSERT_TRUE(ni.has_uninitialized_rank());
}

TEST_F(TypesTest, convertsToPair) {
    const auto ni = NodeIdentifier{ mpi_rank_type{ 3 }, node_id_type{ 7 } };

    const std::pair<mpi_rank_type, node_id_type> as_pair = ni;

    ASSERT_EQ(as_pair.first, mpi_rank_type{ 3 });
    ASSERT_EQ(as_pair.second, node_id_type{ 7 });
}

TEST_F(TypesTest, convertsToTuple) {
    const auto ni = NodeIdentifier{ mpi_rank_type{ 4 }, node_id_type{ 9 } };

    const std::tuple<mpi_rank_type, node_id_type> as_tuple = ni;

    ASSERT_EQ(std::get<0>(as_tuple), mpi_rank_type{ 4 });
    ASSERT_EQ(std::get<1>(as_tuple), node_id_type{ 9 });
}
