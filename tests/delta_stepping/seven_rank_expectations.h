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

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/paths/DeltaStepping.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <vector>

/**
 * @brief The expected single-source result of one node of a seven-rank test graph.
 */
struct SevenRankSsspExpectation {
    distance_type sum_of_distances{};
    distance_type maximum_distance{};
    double efficiency{};
    global_node_id_type number_unreachable_nodes{};
};

namespace seven_rank_expectations {

/**
 * @brief Appends the expected results of one ten-node block to the list.
 * @param expectations The list to append to
 * @param sums The expected sums of distances, one per node of the block
 * @param maxima The expected eccentricities, one per node of the block
 * @param efficiencies The expected sums of efficiencies, one per node of the block
 * @param unreachable The expected number of unreachable nodes, one per node of the block
 */
inline void append_block(std::vector<SevenRankSsspExpectation>& expectations, const std::vector<distance_type>& sums,
                         const std::vector<distance_type>& maxima, const std::vector<double>& efficiencies,
                         const std::vector<global_node_id_type>& unreachable) {
    for (auto index = std::size_t{ 0 }; index < sums.size(); ++index) {
        expectations.push_back(SevenRankSsspExpectation{ sums[index], maxima[index], efficiencies[index], unreachable[index] });
    }
}

/**
 * @brief Appends the same expected result <count> times, which covers the full graph, where every
 *		node sees exactly the same distances.
 * @param expectations The list to append to
 * @param count How often the result is appended
 * @param sum The expected sum of distances
 * @param maximum The expected eccentricity
 * @param efficiency The expected sum of efficiencies
 * @param unreachable The expected number of unreachable nodes
 */
inline void append_uniform(std::vector<SevenRankSsspExpectation>& expectations, const std::size_t count, const distance_type sum,
                           const distance_type maximum, const double efficiency, const global_node_id_type unreachable) {
    expectations.insert(expectations.end(), count, SevenRankSsspExpectation{ sum, maximum, efficiency, unreachable });
}

/**
 * @brief The expected results of the source/sink block, i.e. of rank 3. Its nodes 0 to 2 are isolated
 *		and its nodes 3 to 5 are sinks, so they only reach themselves; its nodes 6 to 9 reach exactly
 *		their single target over an arc of weight 1, 2, 3 and 4.
 * @param expectations The list to append to
 * @param number_nodes The number of nodes in the whole graph, which fixes the unreachable counts
 */
inline void append_source_sink_block(std::vector<SevenRankSsspExpectation>& expectations, const global_node_id_type number_nodes) {
    append_uniform(expectations, 6, 0, 0, 0.0, number_nodes - 1);
    expectations.push_back(SevenRankSsspExpectation{ 1, 1, 1.0, number_nodes - 2 });
    expectations.push_back(SevenRankSsspExpectation{ 2, 2, 0.5, number_nodes - 2 });
    expectations.push_back(SevenRankSsspExpectation{ 3, 3, 1.0 / 3.0, number_nodes - 2 });
    expectations.push_back(SevenRankSsspExpectation{ 4, 4, 0.25, number_nodes - 2 });
}

} // namespace seven_rank_expectations

/**
 * @brief The expected per-node results of get_standard_seven_rank_graph(), in the order of the
 *		flattened graph. The four blocks of the large ring see the same distances as in the four-rank
 *		graph and reach 40 of the 70 nodes; the two blocks of the small ring reach 20.
 * @return One expectation per node
 */
[[nodiscard]] inline std::vector<SevenRankSsspExpectation> get_standard_seven_rank_expectations() {
    using namespace seven_rank_expectations;

    const auto large_sums = std::vector<distance_type>{ 760, 784, 756, 748, 740, 840, 788, 784, 752, 764 };
    const auto large_maxima = std::vector<distance_type>{ 38, 39, 37, 37, 37, 40, 39, 38, 37, 38 };
    const auto large_efficiencies = std::vector<double>{ 5.8086783847071217, 4.8829981275875758, 5.4463030539884745, 6.1299867303575732, 6.3949926703635134,
                                                         3.4097103354784561, 4.6742945932261470, 4.7133019123031259, 5.5366875043729253, 5.2153791587224756 };
    const auto large_unreachable = std::vector<global_node_id_type>(10, 30);

    const auto small_sums = std::vector<distance_type>{ 160, 172, 158, 154, 150, 200, 174, 172, 156, 162 };
    const auto small_maxima = std::vector<distance_type>{ 16, 17, 15, 15, 15, 18, 17, 16, 15, 16 };
    const auto small_efficiencies = std::vector<double>{ 5.1167790542790522, 4.2047692503574865, 4.7526307026306999, 5.4312021312021292, 5.6914585414585390,
                                                         2.7620665282429986, 3.9986795067677430, 4.0356893106893117, 4.8407259407259392, 4.5263028638028633 };
    const auto small_unreachable = std::vector<global_node_id_type>(10, 50);

    auto expectations = std::vector<SevenRankSsspExpectation>{};
    expectations.reserve(70);

    // The ranks 0, 1, 4 and 6 form the large ring, the ranks 2 and 5 the small one, rank 3 is special
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, small_sums, small_maxima, small_efficiencies, small_unreachable);
    append_source_sink_block(expectations, 70);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, small_sums, small_maxima, small_efficiencies, small_unreachable);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);

    return expectations;
}

/**
 * @brief The expected per-node results of get_standard_uu_seven_rank_graph(). Symmetrizing joins the
 *		sinks and sources of rank 3 into the components {3, 6, 9}, {4, 7} and {5, 8}.
 * @return One expectation per node
 */
[[nodiscard]] inline std::vector<SevenRankSsspExpectation> get_standard_uu_seven_rank_expectations() {
    using namespace seven_rank_expectations;

    const auto large_sums = std::vector<distance_type>{ 104, 92, 104, 92, 92, 100, 92, 108, 96, 96 };
    const auto large_maxima = std::vector<distance_type>{ 5, 4, 5, 4, 4, 4, 4, 5, 4, 5 };
    const auto large_efficiencies = std::vector<double>{ 18.150000000000002, 19.833333333333336, 17.616666666666667, 19.833333333333332, 19.833333333333332,
                                                         18.0, 19.833333333333336, 17.233333333333334, 18.916666666666668, 19.449999999999999 };
    const auto large_unreachable = std::vector<global_node_id_type>(10, 30);

    const auto small_sums = std::vector<distance_type>{ 42, 36, 42, 36, 36, 40, 36, 44, 38, 38 };
    const auto small_maxima = std::vector<distance_type>{ 4, 3, 4, 3, 3, 3, 3, 4, 3, 4 };
    const auto small_efficiencies = std::vector<double>{ 10.666666666666668, 11.833333333333336, 10.250000000000002, 11.833333333333334, 11.833333333333334,
                                                         10.5, 11.833333333333336, 10.0, 11.166666666666668, 11.583333333333334 };
    const auto small_unreachable = std::vector<global_node_id_type>(10, 50);

    // The symmetrized source/sink block: three isolated nodes, then {3, 6, 9}, {4, 7} and {5, 8}
    const auto rank_three_sums = std::vector<distance_type>{ 0, 0, 0, 2, 1, 1, 3, 1, 1, 3 };
    const auto rank_three_maxima = std::vector<distance_type>{ 0, 0, 0, 1, 1, 1, 2, 1, 1, 2 };
    const auto rank_three_efficiencies = std::vector<double>{ 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5 };
    const auto rank_three_unreachable = std::vector<global_node_id_type>{ 69, 69, 69, 67, 68, 68, 67, 68, 68, 67 };

    auto expectations = std::vector<SevenRankSsspExpectation>{};
    expectations.reserve(70);

    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, small_sums, small_maxima, small_efficiencies, small_unreachable);
    append_block(expectations, rank_three_sums, rank_three_maxima, rank_three_efficiencies, rank_three_unreachable);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);
    append_block(expectations, small_sums, small_maxima, small_efficiencies, small_unreachable);
    append_block(expectations, large_sums, large_maxima, large_efficiencies, large_unreachable);

    return expectations;
}

/**
 * @brief The expected per-node results of get_full_seven_rank_graph(). The graph is complete, so
 *		every node reaches each of the 62 others over a single arc of weight one.
 * @return One expectation per node
 */
[[nodiscard]] inline std::vector<SevenRankSsspExpectation> get_full_seven_rank_expectations() {
    using namespace seven_rank_expectations;

    auto expectations = std::vector<SevenRankSsspExpectation>{};
    expectations.reserve(63);

    append_uniform(expectations, 63, 62, 1, 62.0, 0);

    return expectations;
}

/**
 * @brief The nodes of a seven-rank graph as (rank, local id), in the order in which the flattened
 *		version numbers them. On one rank every node belongs to rank 0 and keeps its flattened id.
 * @param node_distribution The number of nodes on each of the seven ranks
 * @return One identifier per node
 */
[[nodiscard]] inline std::vector<NodeIdentifier> get_seven_rank_identifiers(const std::vector<node_id_type>& node_distribution) {
    auto identifiers = std::vector<NodeIdentifier>{};

    if (mpiPP::MPIInfo::get_number_ranks() == 1) {
        auto number_nodes = node_id_type{ 0 };
        for (const auto count : node_distribution) {
            number_nodes += count;
        }
        for (auto node_id = node_id_type{ 0 }; node_id < number_nodes; ++node_id) {
            identifiers.emplace_back(mpi_rank_type{ 0 }, node_id);
        }
        return identifiers;
    }

    for (auto rank = mpi_rank_type{ 0 }; rank < static_cast<mpi_rank_type>(node_distribution.size()); ++rank) {
        for (auto node_id = node_id_type{ 0 }; node_id < node_distribution[static_cast<std::size_t>(rank)]; ++node_id) {
            identifiers.emplace_back(rank, node_id);
        }
    }

    return identifiers;
}

/**
 * @brief Runs one delta-stepping computation per node of the graph and compares the four returned
 *		values against the expectations. Every rank must enter this, but only the root rank holds the
 *		reduced result.
 * @param graph The graph
 * @param node_distribution The number of nodes on each of the seven ranks
 * @param expectations One expectation per node, in the order of the flattened graph
 * @param delta The delta of the computation
 * @param epoch_type The epoch strategy under test
 */
inline void expect_seven_rank_delta_stepping(const DistributedGraph& graph, const std::vector<node_id_type>& node_distribution,
                                             const std::vector<SevenRankSsspExpectation>& expectations, const distance_type delta,
                                             const DeltaSteppingEpochType epoch_type) {
    const auto identifiers = get_seven_rank_identifiers(node_distribution);
    ASSERT_EQ(identifiers.size(), expectations.size());

    for (auto index = std::size_t{ 0 }; index < identifiers.size(); ++index) {
        const auto& [sum_of_distances, maximum_distance, efficiency, number_unreachable_nodes] = DeltaStepping::compute_shortest_distances(graph, identifiers[index], delta, epoch_type);

        if (!mpiPP::MPIInfo::is_root_rank()) {
            continue;
        }

        const auto& expected = expectations[index];
        ASSERT_EQ(sum_of_distances, expected.sum_of_distances) << "at the node " << index;
        ASSERT_EQ(maximum_distance, expected.maximum_distance) << "at the node " << index;
        ASSERT_NEAR(efficiency, expected.efficiency, 1e-6) << "at the node " << index;
        ASSERT_EQ(number_unreachable_nodes, expected.number_unreachable_nodes) << "at the node " << index;
    }
}

/// The number of nodes each of the seven ranks owns in the standard and in the full graph
inline const auto standard_seven_rank_distribution = std::vector<node_id_type>{ 10, 10, 10, 10, 10, 10, 10 };
inline const auto full_seven_rank_distribution = std::vector<node_id_type>{ 8, 9, 8, 10, 9, 9, 10 };
