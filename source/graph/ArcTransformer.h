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

#include "graph/GraphTypes.h"

/**
 * @brief This class offers functionality to transform arcs, e.g., by setting all weights to one.
 */
class ArcTransformer {
public:
    using Arcs = LoadedArcs;

    /**
     * @brief Removes all arcs that connect a node to itself
     * @param arcs The arcs to transform
     * @param my_rank The current MPI rank
     * @return The input arcs without self arcs
     */
    [[nodiscard]] static Arcs remove_self_arcs(Arcs arcs, mpi_rank_type my_rank);

    /**
     * @brief Removes all arcs which have weight zero
     * @param arcs The arcs to transform
     * @return The input arcs without weightless arcs
     */
    [[nodiscard]] static Arcs remove_weight_zero(Arcs arcs);

    /**
     * @brief Makes all weights on the arcs positive
     * @param arcs The arcs to transform
     * @return The input arcs with positive weights
     */
    [[nodiscard]] static Arcs all_weights_positive(Arcs arcs);

    /**
     * @brief Makes all weights on the arcs one
     * @param arcs The arcs to transform
     * @return The input arcs with unit weight
     */
    [[nodiscard]] static Arcs all_weights_one(Arcs arcs);

    /**
     * @brief Synchronizes the arcs, i.e., all incoming arcs will be outgoing arcs and vice versa.
     *		Adds the weights; if the added weight is zero, sets it to one.
     * @param incoming_arcs The incoming arcs
     * @param outgoing_arcs The outgoing arcs
     * @return The synchronized arcs. As incoming and outgoing arcs are now the same, returns only one value
     */
    [[nodiscard]] static Arcs synchronize_arcs(Arcs incoming_arcs, Arcs outgoing_arcs);
};
