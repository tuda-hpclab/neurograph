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

#include "GraphTest.h"

#include "Types.h"

#include <gtest/gtest.h>

/**
 * @brief Checks AllPairsShortestPath::compute_apsp_delta_stepping, i.e., the all-pairs computation
 *		that repeats a delta-stepping single-source run instead of a Dijkstra one, for every
 *		DeltaSteppingEpochType.
 */
class APSPDeltaSteppingTest : public GraphTest, public testing::WithParamInterface<DeltaSteppingEpochType> { };
