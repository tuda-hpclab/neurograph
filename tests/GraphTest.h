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

#include <gtest/gtest.h>

class GraphTest : public ::testing::Test {
protected:
    [[nodiscard]] static DistributedGraph get_standard_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_empty_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_uu_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_small_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_full_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_negative_weight_one_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_uu_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_full_four_rank_graph();

    /**
     * @brief A graph distributed over seven ranks and split into three independent parts:
     *        ranks 0, 1, 4, and 6 form the standard four-rank graph; ranks 2 and 5 form
     *        its two-rank version; rank 3 contains isolated nodes, sinks, and sources.
     * @return The graph, distributed over seven ranks
     */
    [[nodiscard]] static DistributedGraph get_standard_seven_rank_graph();

    /**
     * @brief The undirected, unit-weight version of get_standard_seven_rank_graph().
     * @return The graph, distributed over seven ranks
     */
    [[nodiscard]] static DistributedGraph get_standard_uu_seven_rank_graph();

    /**
     * @brief The seven-rank graph with complete four-rank and two-rank components;
     *        rank 3 still contains isolated nodes, sinks, and sources.
     * @return The graph, distributed over seven ranks
     */
    [[nodiscard]] static DistributedGraph get_full_seven_rank_graph();

    /**
     * @brief A graph whose twelve nodes (three per rank) form seven strongly connected components:
     *		one of four nodes that runs through all four ranks, two of two nodes (one within rank 1, one
     *		across ranks 2 and 3), and four single nodes, connected by arcs that lie on no cycle.
     *		Unlike get_small_four_rank_graph, the in arcs mirror the out arcs.
     * @return The graph, distributed over four ranks
     */
    [[nodiscard]] static DistributedGraph get_multi_component_four_rank_graph();

    [[nodiscard]] static DistributedGraph get_standard_four_rank_graph_on_one_rank();

    [[nodiscard]] static DistributedGraph get_standard_uu_four_rank_graph_on_one_rank();

    [[nodiscard]] static DistributedGraph get_full_four_rank_graph_on_one_rank();

    /**
     * @brief The graph of get_standard_seven_rank_graph() flattened onto one rank.
     * @return The graph, on one rank
     */
    [[nodiscard]] static DistributedGraph get_standard_seven_rank_graph_on_one_rank();

    /**
     * @brief The graph of get_standard_uu_seven_rank_graph() flattened onto one rank.
     * @return The graph, on one rank
     */
    [[nodiscard]] static DistributedGraph get_standard_uu_seven_rank_graph_on_one_rank();

    /**
     * @brief The graph of get_full_seven_rank_graph() flattened onto one rank.
     * @return The graph, on one rank
     */
    [[nodiscard]] static DistributedGraph get_full_seven_rank_graph_on_one_rank();

    /**
     * @brief The graph of get_multi_component_four_rank_graph on a single rank, where the node (rank, id)
     *		became the node 3 * rank + id
     * @return The graph, on one rank
     */
    [[nodiscard]] static DistributedGraph get_multi_component_four_rank_graph_on_one_rank();

    /**
     * @brief Checks whether the current test has to be skipped because it is run
     *        with a number of ranks it does not support. When skipping, the root
     *        rank prints a short note explaining why.
     * @param required_number_ranks The number of ranks the calling test needs
     * @return true iff the test should return early (wrong number of ranks)
     */
    [[nodiscard]] static bool skip_unless_rank_count(int required_number_ranks);

    void SetUp() override; // Called immediately after the constructor for each test

    void TearDown() override; // Called immediately before the destructor for each test
};
