/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Transitivity.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/local_structure/TriangleCounting.h"

#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <cstdint>
#include <utility>

std::pair<double, double> Transitivity::compute_transitivity(const DistributedGraph& graph) {
    const auto number_local_nodes = graph.get_number_local_nodes();

    auto status = mpiPP::MPIProgress{ number_local_nodes, "Transitivity" };

    auto local_closed_triangles = std::uint64_t{ 0 };
    auto local_all_triangles = std::uint64_t{ 0 };

    auto local_closed_triangles_2 = std::uint64_t{ 0 };
    auto local_all_triangles_2 = std::uint64_t{ 0 };

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; node_id++) {
        // Both variants are counted in the same traversal, because they share the neighborhoods that
        // the graph caches for the nodes of the other MPI ranks
        const auto [closed_triangles, all_triangles] = TriangleCounting::count_triangles_node(graph, node_id);
        local_closed_triangles += closed_triangles;
        local_all_triangles += all_triangles;

        const auto [closed_triangles_2, all_triangles_2] = TriangleCounting::count_triangles_node_2(graph, node_id);
        local_closed_triangles_2 += closed_triangles_2;
        local_all_triangles_2 += all_triangles_2;

        status.report(node_id);
    }

    status.finish();

    const auto closed_triangles = mpiPP::MPIReductions::reduce_sum(local_closed_triangles);
    const auto all_triangles = mpiPP::MPIReductions::reduce_sum(local_all_triangles);

    const auto closed_triangles_2 = mpiPP::MPIReductions::reduce_sum(local_closed_triangles_2);
    const auto all_triangles_2 = mpiPP::MPIReductions::reduce_sum(local_all_triangles_2);

    // No node has a triangle it could close, which also covers every rank but the root one, where the
    // reductions above return zero
    const auto transitivity = all_triangles == 0 ? 0.0 : static_cast<double>(closed_triangles) / static_cast<double>(all_triangles);
    const auto transitivity_2 = all_triangles_2 == 0 ? 0.0 : static_cast<double>(closed_triangles_2) / static_cast<double>(all_triangles_2);

    return { transitivity, transitivity_2 };
}
