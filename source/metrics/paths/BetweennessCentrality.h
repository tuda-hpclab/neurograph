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

#include "graph/DistributedGraph.h"
#include "metrics/counting/NodeCounter.h"
#include "metrics/counting/NodeDistributionCounter.h"
#include "metrics/paths/BetweennessCentralityVariants/BrandesVariant.h"
#include "metrics/paths/BetweennessCentralityVariants/PathEnumerationVariant.h"

#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/prefix_sum.hpp>

#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <functional>
#include <numeric>
#include <span>
#include <vector>

class BetweennessCentrality {
public:
    /**
     * @brief Calculates the average betweenness centrality in the graph, i.e., for all nodes i and j,
     *		looks at all nodes on the paths p1, ..., pk from i to j with minimal length.
     *		Dispatches to the algorithm selected by algorithm; all algorithms compute identical
     *		values, but their complexity differs (see the types in metrics/paths/BetweennessCentralityVariants).
     *		Must be called on every MPI rank, returns the result only on MPI rank 0.
     * @param graph The graph
     * @param algorithm Which betweenness centrality algorithm to use
     * @return The betweenness centrality averaged across all nodes in the network
     */
    [[nodiscard]] static double compute_average_betweenness_centrality(const DistributedGraph& graph,
                                                                       const BetweennessCentralityAlgorithm algorithm = BetweennessCentralityAlgorithm::Brandes) {
        const auto number_local_nodes = graph.get_number_local_nodes();
        const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
        const auto node_distribution_span = std::span{ node_distribution };
        const auto prefix_distribution = utility::calculate_prefix_sum(node_distribution_span);

        auto status = mpiPP::MPIProgress{ number_local_nodes, "Betweenness Centrality" };

        auto local_betweenness = std::vector<double>(total_number_nodes, 0.0);

        switch (algorithm) {
        case BetweennessCentralityAlgorithm::Brandes:
            BrandesVariant::accumulate_betweenness(graph, prefix_distribution, local_betweenness, status);
            break;
        case BetweennessCentralityAlgorithm::PathEnumeration:
            PathEnumerationVariant::accumulate_betweenness(graph, prefix_distribution, local_betweenness, status);
            break;
        default:
            utility::Exception::fail("BetweennessCentrality::compute_average_betweenness_centrality: the algorithm is unknown");
        }

        mpiPP::MPISynchronization::barrier();

        status.finish();

        const auto result = mpiPP::MPIReductions::reduce_componentwise_sum(local_betweenness);

        const auto betweenness_centrality = std::reduce(result.begin(), result.end(), 0.0, std::plus<double>{});
        const auto average_betweenness_centrality = betweenness_centrality / static_cast<double>(total_number_nodes);

        return average_betweenness_centrality;
    }
};
