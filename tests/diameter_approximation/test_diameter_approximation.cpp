/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_diameter_approximation.h"

#include "Types.h"

#include "metrics/paths/DiameterApproximation.h"
#include "utility/RandomNumberGenerator.h"

#include <mpi-wrapper/core/MPIInfo.h>

namespace {
/**
 * @brief Always returns the same value. With 0.0 every node passes the center threshold in the very
 *		first iteration, so every node becomes its own center and the contracted graph is the
 *		original graph: the approximation then returns the exact diameter plus 2 * r.
 */
class ConstantRandomNumberGenerator : public RandomNumberGenerator {
public:
    explicit ConstantRandomNumberGenerator(const double constant_value)
        : value(constant_value) { }

    [[nodiscard]] double draw() override {
        return value;
    }

private:
    double value{};
};

/**
 * @brief Returns 0.0 on the first draw and 1.0 afterwards, so on every rank exactly the first node
 *		that draws (the first local node without a center) becomes a center. On one rank the whole
 *		graph is contracted into that single center and the approximation returns 0 + 2 * r.
 */
class SingleCenterRandomNumberGenerator : public RandomNumberGenerator {
public:
    [[nodiscard]] double draw() override {
        if (first_draw) {
            first_draw = false;
            return 0.0;
        }

        return 1.0;
    }

private:
    bool first_draw{ true };
};
} // namespace

TEST_F(DiameterApproximationTest, testAllCentersStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 7 (see the APSP tests): 7 + 2 * 2 = 11
    ASSERT_EQ(approximated_diameter, 11);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 1, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 3 (see the APSP tests): 3 + 2 * 1 = 5
    ASSERT_EQ(approximated_diameter, 5);
}

TEST_F(DiameterApproximationTest, testAllCentersFull) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_one_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 1 (see the APSP tests): 1 + 2 * 2 = 5
    ASSERT_EQ(approximated_diameter, 5);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 40 (see the APSP tests): 40 + 2 * 2 = 44
    ASSERT_EQ(approximated_diameter, 44);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardUUFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 5 (see the APSP tests): 5 + 2 * 2 = 9
    ASSERT_EQ(approximated_diameter, 9);
}

TEST_F(DiameterApproximationTest, testAllCentersFullFourRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_four_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 1 (see the APSP tests): 1 + 2 * 2 = 5
    ASSERT_EQ(approximated_diameter, 5);
}

TEST_F(DiameterApproximationTest, testSingleCenterStandard) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_one_rank_graph();

    auto generator = SingleCenterRandomNumberGenerator{};
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Node 0 becomes the only center. Its ball reaches every node with distance at most 2 * r = 4
    // in iteration 0 and node 5 (distance 5, within the grown threshold of 8) in iteration 1, so
    // the whole graph is contracted into a single node: 0 + 2 * 2 = 4
    ASSERT_EQ(approximated_diameter, 4);
}

TEST_F(DiameterApproximationTest, testSingleCenterStandardUU) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    auto generator = SingleCenterRandomNumberGenerator{};
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 1, generator);

    // Node 0 becomes the only center. Every node is within distance 3 of it (see the APSP tests),
    // so the ball swallows the whole graph after two growing steps and the contracted graph is a
    // single node: 0 + 2 * 1 = 2
    ASSERT_EQ(approximated_diameter, 2);
}

TEST_F(DiameterApproximationTest, testDefaultGenerator) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_one_rank_graph();

    // The default generator is seeded with the MPI rank, so two runs must choose the same centers
    // and therefore compute the same approximation
    const auto first_approximation = DiameterApproximation::compute_approximation(graph, 2);
    const auto second_approximation = DiameterApproximation::compute_approximation(graph, 2);

    ASSERT_EQ(first_approximation, second_approximation);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_four_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // Every node is its own center, so the contracted graph is the original graph whose exact
        // diameter is 40 (see the APSP tests): 40 + 2 * 2 = 44
        ASSERT_EQ(approximated_diameter, 44);
    }
}

TEST_F(DiameterApproximationTest, testAllCentersStandardUUFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // Every node is its own center, so the contracted graph is the original graph whose exact
        // diameter is 5 (see the APSP tests): 5 + 2 * 2 = 9
        ASSERT_EQ(approximated_diameter, 9);
    }
}

TEST_F(DiameterApproximationTest, testAllCentersFullFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_full_four_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // Every node is its own center, so the contracted graph is the original graph whose exact
        // diameter is 1 (see the APSP tests): 1 + 2 * 2 = 5
        ASSERT_EQ(approximated_diameter, 5);
    }
}

TEST_F(DiameterApproximationTest, testDefaultGeneratorFourRanks) {
    if (skip_unless_rank_count(4)) {
        return;
    }

    const auto graph = get_standard_uu_four_rank_graph();

    // The default generator is seeded with the MPI rank, so two runs must choose the same centers
    // and therefore compute the same approximation
    const auto first_approximation = DiameterApproximation::compute_approximation(graph, 2);
    const auto second_approximation = DiameterApproximation::compute_approximation(graph, 2);

    if (mpiPP::MPIInfo::is_root_rank()) {
        ASSERT_EQ(first_approximation, second_approximation);
    }
}

TEST_F(DiameterApproximationTest, testAllCentersStandardSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 40 (see the APSP tests): 40 + 2 * 2 = 44. The graph being disconnected does not
    // matter here, because the diameter only looks at the pairs that are reachable at all.
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(approximated_diameter, 44);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardUUSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph whose exact
    // diameter is 5 (see the APSP tests): 5 + 2 * 2 = 9
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(approximated_diameter, 9);
}

TEST_F(DiameterApproximationTest, testAllCentersFullSevenRanksDummy) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph_on_one_rank();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    // Every node is its own center, so the contracted graph is the original graph. In the complete
    // graph every node is one unit-weight arc away from every other one: 1 + 2 * 2 = 5
    // The value serves as standard for the seven-rank test
    ASSERT_EQ(approximated_diameter, 5);
}

TEST_F(DiameterApproximationTest, testAllCentersStandardSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_seven_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_EQ(approximated_diameter, 44);
    }
}

TEST_F(DiameterApproximationTest, testAllCentersStandardUUSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_standard_uu_seven_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_EQ(approximated_diameter, 9);
    }
}

TEST_F(DiameterApproximationTest, testAllCentersFullSevenRanks) {
    if (skip_unless_rank_count(7)) {
        return;
    }

    const auto graph = get_full_seven_rank_graph();

    auto generator = ConstantRandomNumberGenerator{ 0.0 };
    const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2, generator);

    if (mpiPP::MPIInfo::is_root_rank()) {
        // The value is the corresponding one from the one-rank version
        ASSERT_EQ(approximated_diameter, 5);
    }
}
