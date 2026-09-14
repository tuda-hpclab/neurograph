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

/**
 * @brief This class represents an in arc, i.e., a directed edge that starts at a node identified by the member variables.
 *		The end is usually identified by the position in a vector and the current MPI rank.
 */
struct InArc {
    mpi_rank_type source_rank{};
    node_id_type source_id{};
    weight_type weight{};

    [[nodiscard]] friend auto operator<=>(const InArc&, const InArc&) = default;

    friend std::ostream& operator<<(std::ostream& out, const InArc& arc) {
        out << '(' << arc.source_rank << ", " << arc.source_id << ": " << arc.weight << ')';
        return out;
    }
};

/**
 * @brief This class represents an out arc, i.e., a directed edge that ends at a node identified by the member variables.
 *		The start is usually identified by the position in a vector and the current MPI rank.
 */
struct OutArc {
    mpi_rank_type target_rank{};
    node_id_type target_id{};
    weight_type weight{};

    [[nodiscard]] friend auto operator<=>(const OutArc&, const OutArc&) = default;

    friend std::ostream& operator<<(std::ostream& out, const OutArc& arc) {
        out << '(' << arc.target_rank << ", " << arc.target_id << ": " << arc.weight << ')';
        return out;
    }
};
