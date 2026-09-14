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

/**
 * @brief Checks NetworkMotifAlgorithm::RmaFair, i.e., the RMA variant that spreads the triangles spanning three MPI ranks over their three owners.
 */
class NetworkMotifsRmaFairTest : public GraphTest { };
