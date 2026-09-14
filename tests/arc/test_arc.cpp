/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_arc.h"

#include "Types.h"

#include "graph/Arc.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST_F(ArcTest, inArcStreamsRankIdAndWeight) {
    const auto arc = InArc{ mpi_rank_type{ 2 }, node_id_type{ 5 }, weight_type{ -3 } };

    auto out = std::ostringstream{};
    out << arc;

    ASSERT_EQ(out.str(), "(2, 5: -3)");
}

TEST_F(ArcTest, outArcStreamsRankIdAndWeight) {
    const auto arc = OutArc{ mpi_rank_type{ 1 }, node_id_type{ 7 }, weight_type{ 4 } };

    auto out = std::ostringstream{};
    out << arc;

    ASSERT_EQ(out.str(), "(1, 7: 4)");
}

TEST_F(ArcTest, defaultConstructedArcsAreZeroInitialized) {
    const auto in_arc = InArc{};
    ASSERT_EQ(in_arc.source_rank, mpi_rank_type{ 0 });
    ASSERT_EQ(in_arc.source_id, node_id_type{ 0 });
    ASSERT_EQ(in_arc.weight, weight_type{ 0 });

    const auto out_arc = OutArc{};
    ASSERT_EQ(out_arc.target_rank, mpi_rank_type{ 0 });
    ASSERT_EQ(out_arc.target_id, node_id_type{ 0 });
    ASSERT_EQ(out_arc.weight, weight_type{ 0 });

    // exercise the defaulted three-way comparison
    ASSERT_TRUE(in_arc == InArc{});
    ASSERT_TRUE(out_arc == OutArc{});
}
