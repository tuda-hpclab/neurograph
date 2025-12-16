#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "graph/DistributedGraph.h"

#include <gtest/gtest.h>

class GraphTest : public ::testing::Test {
protected:
    [[nodiscard]] static DistributedGraph get_standard_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_uu_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_small_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_full_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_uu_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_full_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_four_rank_graph_on_one_rank();

    [[nodiscard]] static DistributedGraph get_standard_uu_four_rank_graph_on_one_rank();

    [[nodiscard]] static DistributedGraph get_full_four_rank_graph_on_one_rank();

    void SetUp() override; // Called immediately after the constructor for each test

    void TearDown() override; // Called immediately before the destructor for each test
};
