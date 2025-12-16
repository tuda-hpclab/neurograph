/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_diameter_approximation.h"

#include "Types.h"

#include "metrics/AllPairsShortestPath.h"
#include "metrics/DiameterApproximation.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

// TEST_F(DiameterApproximationTest, testStandard) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_one_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2);
//
//	// ASSERT_EQ(approximated_diameter, 9);
// }
//
// TEST_F(DiameterApproximationTest, testStandardUU) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_uu_one_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2);
//
//	// ASSERT_EQ(approximated_diameter, 5);
// }
//
// TEST_F(DiameterApproximationTest, testFull) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_full_one_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2);
//
//	// ASSERT_EQ(approximated_diameter, 2);
// }
//
// TEST_F(DiameterApproximationTest, testStandardFourRanksDummy) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_four_rank_graph_on_one_rank();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 6);
//
//	// ASSERT_EQ(approximated_diameter, 48);
// }
//
// TEST_F(DiameterApproximationTest, testStandardUUFourRanksDummy) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_uu_four_rank_graph_on_one_rank();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 1);
//
//	// ASSERT_EQ(approximated_diameter, 6);
// }
//
// TEST_F(DiameterApproximationTest, testFullFourRanksDummy) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 1) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 1 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_full_four_rank_graph_on_one_rank();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2);
//
//	// ASSERT_EQ(approximated_diameter, 0);
// }
//
// TEST_F(DiameterApproximationTest, testStandardFourRanks) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 4 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_four_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 6);
//
//	if (mpiPP::MPIInfo::is_root_rank()) {
//		// ASSERT_EQ(approximated_diameter, 48);
//	}
// }
//
// TEST_F(DiameterApproximationTest, testStandardUUFourRanks) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 4 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_standard_uu_four_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 1);
//
//	if (mpiPP::MPIInfo::is_root_rank()) {
//		// ASSERT_EQ(approximated_diameter, 6);
//	}
// }
//
// TEST_F(DiameterApproximationTest, testFullFourRanks) {
//	if (mpiPP::MPIInfo::get_number_ranks_cast() != 4) {
//		if (mpiPP::MPIInfo::is_root_rank()) {
//			spdlog::info("Test only works with 4 MPI ranks.");
//		}
//
//		return;
//	}
//
//	const auto graph = get_full_four_rank_graph();
//
//	const auto approximated_diameter = DiameterApproximation::compute_approximation(graph, 2);
//
//	if (mpiPP::MPIInfo::is_root_rank()) {
//		// ASSERT_EQ(approximated_diameter, 0);
//	}
// }
