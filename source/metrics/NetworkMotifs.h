#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include <array>
#include <cstdint>
#include <vector>

class DistributedGraph;

class NetworkMotifs {
public:
    /**
     * Function to compute the fraction of NetworkMotifs of the graph
     *
     * Returns: The fraction of NetworkMotifs in regard to their total count using
     *          the network motif types and numeration according to paper
     *  "A tutorial in connectome analysis: Topological and spatial features of brain networks"
     *  by Marcus Kaiser, 2011 in NeuroImage, (892-907), page 899
     *          result[0] is the total count of motifs
     *          result[i] is the fraction of motif i [1...13] (total count motif i / result[0])
     *
     * Parameters
     * graph:           A DistributedGraph (Function is MPI compliant)
     *
     * MPI Constraints: Function must be called on all ranks simultaneously
     * 					Function returns correct information to all ranks
     */
    [[nodiscard]] static std::array<long double, 14> compute_network_triple_motifs(const DistributedGraph& graph);

    [[nodiscard]] static std::array<long double, 14> compute_network_triple_motifs_sequential(const DistributedGraph& graph);

    struct ThreeMotifStructure {
        mpi_rank_type node_3_rank = 0;
        node_id_type node_3_local = 0;
        std::uint16_t motif_type_bit_array = 0;

        void self_test();
        void set_motif_types(const std::vector<unsigned int>& motif_types);
        void unset_motif_types(const std::vector<unsigned int>& motif_types);
        void unset_all_but_motif_types(const std::vector<unsigned int>& motif_types);
        [[nodiscard]] bool is_motif_type_set(unsigned int motif_type) const;
        void print_out_complete() const;
        void print_out() const;
        [[nodiscard]] bool check_validity() const;
    };

private:
    [[nodiscard]] static std::uint16_t update_arc_bit_array(const DistributedGraph& graph, std::uint16_t exists_arc_bit_array, mpi_rank_type node_2_rank,
                                                            node_id_type node_2_local, mpi_rank_type node_3_rank, node_id_type node_3_local);
};
