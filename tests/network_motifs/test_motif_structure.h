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
 * @brief Checks NetworkMotifs::ThreeMotifStructure, i.e., the bit array in which the algorithms of
 *		NetworkMotifs record which motifs a triple can still be, independently of a graph.
 */
class ThreeMotifStructureTest : public GraphTest { };
