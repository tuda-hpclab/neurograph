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

#include <array>
#include <cstdint>
#include <vector>

class DistributedGraph;

class NetworkMotifs {
public:
    /**
     * @brief Computes the fraction of each three-node network motif in the graph with the selected
     *		algorithm. The motif types and their numbering follow the paper
     *		"A tutorial in connectome analysis: Topological and spatial features of brain networks"
     *		by Marcus Kaiser, 2011 in NeuroImage, (892-907), page 899.
     *		All algorithms are deterministic and compute the same fractions; they differ in how the
     *		arcs between two nodes of a triple that the current MPI rank does not own are obtained
     *		and in how the resulting work is spread over the MPI ranks, see NetworkMotifAlgorithm.
     *		Must be called on every MPI rank with the same algorithm, returns the result only on
     *		MPI rank 0.
     *		Complexity is O(sum over all nodes of the degree squared), neglecting MPI collectives.
     * @param graph The distributed graph
     * @param algorithm Which algorithm to use; Questions is the all-round choice, as it needs no
     *		remote arc reads at all
     * @return An array where result[0] is the total count of motifs and result[i] (i in [1, 13])
     *		is the fraction of motif i, i.e., the count of motif i divided by result[0].
     *		The result is meaningful only on MPI rank 0; every other rank returns an all-zero array.
     */
    [[nodiscard]] static std::array<long double, 14> compute_network_triple_motifs(const DistributedGraph& graph,
                                                                                   NetworkMotifAlgorithm algorithm = NetworkMotifAlgorithm::Questions);

    /**
     * @brief Sequential reference implementation of compute_network_triple_motifs; the whole graph
     *		is traversed on the root rank only, intended for validating the distributed version.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     *		Complexity is O(sum over all nodes of the degree squared), neglecting MPI collectives.
     * @param graph The distributed graph
     * @return The motif fractions in the same layout as compute_network_triple_motifs;
     *		meaningful only on MPI rank 0.
     */
    [[nodiscard]] static std::array<long double, 14> compute_network_triple_motifs_sequential(const DistributedGraph& graph);

    struct ThreeMotifStructure {
        mpi_rank_type node_3_rank = 0;
        node_id_type node_3_local = 0;
        std::uint16_t motif_type_bit_array = 0;

        void self_test();
        void set_motif_types(const std::vector<unsigned int>& motif_types);
        void unset_motif_types(const std::vector<unsigned int>& motif_types);
        [[nodiscard]] bool is_motif_type_set(unsigned int motif_type) const;
        void print_out() const;
        [[nodiscard]] bool check_validity() const;
    };

private:
    /**
     * @brief The implementation of NetworkMotifAlgorithm::Questions, i.e., the variant that asks the
     *		owner of node 2 of a triple whether it is connected to node 3 instead of reading its arcs.
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     * @param graph The distributed graph
     * @return The motif fractions, meaningful only on MPI rank 0
     */
    [[nodiscard]] static std::array<long double, 14> compute_motifs_with_questions(const DistributedGraph& graph);

    [[nodiscard]] static std::uint16_t update_arc_bit_array(const DistributedGraph& graph, std::uint16_t exists_arc_bit_array, mpi_rank_type node_2_rank,
                                                            node_id_type node_2_local, mpi_rank_type node_3_rank, node_id_type node_3_local);
};
