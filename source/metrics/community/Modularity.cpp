/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Modularity.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/community/ModularityVariants/CommunityDetectionCommon.h"
#include "metrics/community/ModularityVariants/CommunityGraph.h"
#include "metrics/community/ModularityVariants/InfomapVariant.h"
#include "metrics/community/ModularityVariants/LabelPropagationVariant.h"
#include "metrics/community/ModularityVariants/LeidenVariant.h"
#include "metrics/community/ModularityVariants/LouvainVariant.h"
#include "metrics/counting/NodeDistributionCounter.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/prefix_sum.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <mpi-wrapper/collectives/MPIBroadcasts.h>
#include <mpi-wrapper/collectives/MPIGatherV.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

double Modularity::compute_modularity(const DistributedGraph& graph, const ModularityAlgorithm algorithm) {
    auto timer_gather_arcs = utility::RegionTimer{ "modularity-gather-arcs" };

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = graph.get_number_local_nodes();

    const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
    const auto prefix_distribution = utility::calculate_prefix_sum(std::span{ node_distribution });

    // One entry per local out arc: (global source id, global target id, weight)
    using GlobalArc = std::tuple<global_node_id_type, global_node_id_type, weight_type>;

    auto local_arcs = std::vector<GlobalArc>{};
    local_arcs.reserve(graph.get_number_local_out_arcs());

    const auto my_prefix = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)];

    for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
        const auto source_global_id = my_prefix + node_id;

        for (const auto& [target_rank, target_id, weight] : graph.get_out_arcs(my_rank, node_id)) {
            const auto target_global_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;
            local_arcs.emplace_back(source_global_id, target_global_id, weight);
        }
    }

    const auto gathered_arcs = mpiPP::MPICollectives::gatherv(std::move(local_arcs), mpiPP::MPIRank::root_rank());

    timer_gather_arcs.stop();

    auto timer_detect_communities = utility::RegionTimer{ "modularity-detect-communities" };

    auto modularity = 0.0;

    if (mpiPP::MPIInfo::is_root_rank()) {
        const auto total_number_nodes = std::reduce(node_distribution.begin(), node_distribution.end(), global_node_id_type{ 0 });

        auto community_graph = CommunityGraph{ utility::safe_cast<std::size_t>(total_number_nodes) };

        for (const auto& arcs_of_rank : gathered_arcs) {
            for (const auto& [source, target, weight] : arcs_of_rank) {
                // Modularity is only defined for positive weights, so they enter by absolute value.
                community_graph.add_arc(utility::safe_cast<std::size_t>(source), utility::safe_cast<std::size_t>(target),
                                        std::abs(static_cast<double>(weight)));
            }
        }

        const auto assignment = [&community_graph, algorithm]() -> std::vector<std::size_t> {
            switch (algorithm) {
            case ModularityAlgorithm::Louvain:
                return LouvainVariant::detect_communities(community_graph);
            case ModularityAlgorithm::Leiden:
                return LeidenVariant::detect_communities(community_graph);
            case ModularityAlgorithm::LabelPropagation:
                return LabelPropagationVariant::detect_communities(community_graph);
            case ModularityAlgorithm::Infomap:
                return InfomapVariant::detect_communities(community_graph);
            default:
                utility::Exception::fail("Modularity::compute_modularity: the algorithm is unknown");
            }
        }();

        modularity = CommunityDetectionCommon::compute_modularity(community_graph, assignment);
    }

    timer_detect_communities.stop();

    auto timer_broadcast = utility::RegionTimer{ "modularity-broadcast" };

    const auto result = mpiPP::MPIBroadcasts::broadcast(modularity, mpiPP::MPIRank::root_rank());

    timer_broadcast.stop();

    mpiPP::MPIRegionTimerReport::reduce_and_print("Modularity");
    utility::RegionTimer::init();

    return result;
}
