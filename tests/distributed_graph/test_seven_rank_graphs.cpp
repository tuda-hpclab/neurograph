/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_distributed_graph.h"

#include "Types.h"

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "utility/Vec3.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <span>
#include <tuple>
#include <vector>

namespace {
template <typename Arc>
[[nodiscard]] bool contains_arc(const std::span<const Arc> arcs, const mpi_rank_type rank, const node_id_type node, const weight_type weight) {
    return std::ranges::any_of(arcs, [rank, node, weight](const auto& arc) {
        if constexpr (std::same_as<Arc, InArc>) {
            return arc.source_rank == rank && arc.source_id == node && arc.weight == weight;
        } else {
            return arc.target_rank == rank && arc.target_id == node && arc.weight == weight;
        }
    });
}

[[nodiscard]] std::pair<mpi_rank_type, mpi_rank_type> get_ring_neighbors(const mpi_rank_type rank) {
    switch (rank) {
    case 0:
        return { 6, 1 };
    case 1:
        return { 0, 4 };
    case 2:
        return { 5, 5 };
    case 4:
        return { 1, 6 };
    case 5:
        return { 2, 2 };
    default:
        return { 4, 0 };
    }
}

[[nodiscard]] node_id_type get_expected_full_node_count(const mpi_rank_type rank) {
    constexpr auto counts = std::array<node_id_type, 7>{ 8, 9, 8, 10, 9, 9, 10 };
    return counts[static_cast<std::size_t>(rank)];
}
} // namespace

TEST_F(DistributedGraphTest, testStandardSevenRankGraphTopology) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    ASSERT_EQ(graph.get_number_local_nodes(), node_id_type{ 10 });

    if (my_rank == 3) {
        for (const auto node : { 0U, 1U, 2U }) {
            EXPECT_EQ(graph.get_number_in_arcs(my_rank, node), arc_id_type{ 0 });
            EXPECT_EQ(graph.get_number_out_arcs(my_rank, node), arc_id_type{ 0 });
        }
        for (const auto node : { 3U, 4U, 5U }) {
            EXPECT_GT(graph.get_number_in_arcs(my_rank, node), arc_id_type{ 0 });
            EXPECT_EQ(graph.get_number_out_arcs(my_rank, node), arc_id_type{ 0 });
        }
        for (const auto node : { 6U, 7U, 8U, 9U }) {
            EXPECT_EQ(graph.get_number_in_arcs(my_rank, node), arc_id_type{ 0 });
            EXPECT_GT(graph.get_number_out_arcs(my_rank, node), arc_id_type{ 0 });
        }
        return;
    }

    const auto [predecessor, successor] = get_ring_neighbors(my_rank);
    for (auto node = node_id_type{ 0 }; node < node_id_type{ 10 }; ++node) {
        EXPECT_TRUE(contains_arc(graph.get_in_arcs(my_rank, node), predecessor, node, weight_type{ 11 }));
        EXPECT_TRUE(contains_arc(graph.get_out_arcs(my_rank, node), successor, node, weight_type{ 11 }));
    }
}

TEST_F(DistributedGraphTest, testStandardUndirectedUnitWeightSevenRankGraph) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    for (auto node = node_id_type{ 0 }; node < node_id_type{ 10 }; ++node) {
        const auto in_arcs = graph.get_in_arcs(my_rank, node);
        const auto out_arcs = graph.get_out_arcs(my_rank, node);
        ASSERT_EQ(in_arcs.size(), out_arcs.size());
        EXPECT_TRUE(std::ranges::all_of(in_arcs, [](const auto& arc) { return arc.weight == weight_type{ 1 }; }));
        EXPECT_TRUE(std::ranges::all_of(out_arcs, [](const auto& arc) { return arc.weight == weight_type{ 1 }; }));

        for (const auto& arc : in_arcs) {
            EXPECT_TRUE(contains_arc(out_arcs, arc.source_rank, arc.source_id, weight_type{ 1 }));
        }
    }
}

TEST_F(DistributedGraphTest, testFullSevenRankGraphTopology) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = get_expected_full_node_count(my_rank);
    ASSERT_EQ(graph.get_number_local_nodes(), number_local_nodes);

    // As in get_full_four_rank_graph(), every one of the 63 nodes is connected to every other one
    for (auto node = node_id_type{ 0 }; node < number_local_nodes; ++node) {
        EXPECT_EQ(graph.get_number_in_arcs(my_rank, node), arc_id_type{ 62 });
        EXPECT_EQ(graph.get_number_out_arcs(my_rank, node), arc_id_type{ 62 });
    }
}

TEST_F(DistributedGraphTest, testStandardSevenRankGraphOnOneRank) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();
    constexpr auto rank = mpi_rank_type{ 0 };
    ASSERT_EQ(graph.get_number_local_nodes(), node_id_type{ 70 });
    EXPECT_EQ(graph.get_number_local_in_arcs(), arc_id_type{ 214 });
    EXPECT_EQ(graph.get_number_local_out_arcs(), arc_id_type{ 214 });

    // Cross-rank arcs use the prefix of the former target rank after flattening.
    EXPECT_TRUE(contains_arc(graph.get_out_arcs(rank, 0), rank, 10, 11));
    EXPECT_TRUE(contains_arc(graph.get_out_arcs(rank, 20), rank, 50, 11));
    EXPECT_TRUE(contains_arc(graph.get_out_arcs(rank, 50), rank, 20, 11));
    EXPECT_TRUE(contains_arc(graph.get_out_arcs(rank, 60), rank, 0, 11));

    for (const auto node : { 30U, 31U, 32U }) {
        EXPECT_EQ(graph.get_number_in_arcs(rank, node), arc_id_type{ 0 });
        EXPECT_EQ(graph.get_number_out_arcs(rank, node), arc_id_type{ 0 });
    }
    for (const auto node : { 33U, 34U, 35U }) {
        EXPECT_GT(graph.get_number_in_arcs(rank, node), arc_id_type{ 0 });
        EXPECT_EQ(graph.get_number_out_arcs(rank, node), arc_id_type{ 0 });
    }
    for (const auto node : { 36U, 37U, 38U, 39U }) {
        EXPECT_EQ(graph.get_number_in_arcs(rank, node), arc_id_type{ 0 });
        EXPECT_GT(graph.get_number_out_arcs(rank, node), arc_id_type{ 0 });
    }

    // Canonical rank 1 stores its signal names in reverse order; flattening remaps the indices.
    EXPECT_EQ(graph.get_node_signal_localID(rank, 10), node_id_type{ 0 });
    EXPECT_EQ(graph.get_node_signal_localID(rank, 13), node_id_type{ 1 });
}

TEST_F(DistributedGraphTest, testStandardUndirectedUnitWeightSevenRankGraphOnOneRank) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();
    constexpr auto rank = mpi_rank_type{ 0 };
    ASSERT_EQ(graph.get_number_local_nodes(), node_id_type{ 70 });

    for (auto node = node_id_type{ 0 }; node < node_id_type{ 70 }; ++node) {
        const auto in_arcs = graph.get_in_arcs(rank, node);
        const auto out_arcs = graph.get_out_arcs(rank, node);
        ASSERT_EQ(in_arcs.size(), out_arcs.size());
        EXPECT_TRUE(std::ranges::all_of(in_arcs, [](const auto& arc) { return arc.weight == weight_type{ 1 }; }));
        EXPECT_TRUE(std::ranges::all_of(out_arcs, [](const auto& arc) { return arc.weight == weight_type{ 1 }; }));
    }
}

TEST_F(DistributedGraphTest, testFullSevenRankGraphOnOneRank) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();
    constexpr auto rank = mpi_rank_type{ 0 };
    ASSERT_EQ(graph.get_number_local_nodes(), node_id_type{ 63 });
    EXPECT_EQ(graph.get_number_local_in_arcs(), arc_id_type{ 3906 });
    EXPECT_EQ(graph.get_number_local_out_arcs(), arc_id_type{ 3906 });

    // 63 * 62 = 3906 arcs, i.e., every node is connected to every other one
    for (auto node = node_id_type{ 0 }; node < node_id_type{ 63 }; ++node) {
        EXPECT_EQ(graph.get_number_in_arcs(rank, node), arc_id_type{ 62 });
        EXPECT_EQ(graph.get_number_out_arcs(rank, node), arc_id_type{ 62 });
    }
}

TEST_F(DistributedGraphTest, testSevenRankGraphsPlaceEveryNodeSomewhereElse) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    // Ranks 2, 5, and 6 repeat the canonical node sets of ranks 0, 1, and 3, but get_position_offset()
    // translates their copies, so no two nodes of a seven-rank graph share a position
    const auto expect_distinct_positions = [](const DistributedGraph& graph) {
        constexpr auto rank = mpi_rank_type{ 0 };
        auto positions = std::vector<Vec3d>{};
        positions.reserve(graph.get_number_local_nodes());
        for (auto node = node_id_type{ 0 }; node < graph.get_number_local_nodes(); ++node) {
            positions.push_back(graph.get_node_position(rank, node));
        }

        const auto as_tuple = [](const Vec3d& position) { return std::tie(position.get_x(), position.get_y(), position.get_z()); };
        std::ranges::sort(positions, [&as_tuple](const auto& first, const auto& second) { return as_tuple(first) < as_tuple(second); });
        EXPECT_EQ(std::ranges::adjacent_find(positions), positions.end());
    };

    expect_distinct_positions(get_standard_seven_rank_graph_on_one_rank());
    expect_distinct_positions(get_full_seven_rank_graph_on_one_rank());
}
