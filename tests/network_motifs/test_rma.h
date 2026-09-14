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
 * @brief Checks NetworkMotifAlgorithm::Rma, i.e., the base RMA variant, which reads the arcs of the other MPI ranks instead of asking for them.
 */
class NetworkMotifsRmaTest : public GraphTest { };
