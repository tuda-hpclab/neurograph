/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MaximumFlow.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/counting/NodeDistributionCounter.h"
#include "metrics/flow/MaxFlowVariants/DinicVariant.h"
#include "metrics/flow/MaxFlowVariants/DistributedPushRelabelVariant.h"
#include "metrics/flow/MaxFlowVariants/EdmondsKarpVariant.h"
#include "metrics/flow/MaxFlowVariants/FlowNetwork.h"
#include "metrics/flow/MaxFlowVariants/FordFulkersonVariant.h"
#include "metrics/flow/MaxFlowVariants/OrlinKingRaoTarjanVariant.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/prefix_sum.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <mpi-wrapper/collectives/MPIBroadcasts.h>
#include <mpi-wrapper/collectives/MPIGatherV.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>

#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <span>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
/**
 * @brief Translates the nodes to their global ids, validating every entry against the node
 *		distribution. Duplicates count once; the returned ids keep the order of their first occurrence.
 * @param nodes The nodes as (rank, node id) identifiers
 * @param node_distribution The number of nodes on each MPI rank
 * @param prefix_distribution The prefix sum of node_distribution
 * @param role Either "source" or "sink", only used in error messages
 * @return The distinct global ids in first-occurrence order
 */
[[nodiscard]] std::vector<std::size_t> translate_to_global_ids(const std::span<const NodeIdentifier> nodes,
                                                               const std::span<const global_node_id_type> node_distribution,
                                                               const std::span<const global_node_id_type> prefix_distribution, const char* role) {
    auto global_ids = std::vector<std::size_t>{};
    global_ids.reserve(nodes.size());

    auto seen = std::unordered_set<std::size_t>{};

    for (const auto& [rank, node_id] : nodes) {
        utility::Exception::check(rank >= 0 && utility::safe_cast<std::size_t>(rank) < node_distribution.size(),
                                  "MaximumFlow::compute_maximum_flow: the {} rank {} does not exist", role, rank);
        utility::Exception::check(node_id < node_distribution[utility::safe_cast<std::size_t>(rank)],
                                  "MaximumFlow::compute_maximum_flow: the {} node id {} is out of range, rank {} owns {} nodes", role, node_id, rank,
                                  node_distribution[utility::safe_cast<std::size_t>(rank)]);

        const auto global_id = utility::safe_cast<std::size_t>(prefix_distribution[utility::safe_cast<std::size_t>(rank)] + node_id);

        if (seen.insert(global_id).second) {
            global_ids.push_back(global_id);
        }
    }

    return global_ids;
}
} // namespace

flow_type MaximumFlow::compute_maximum_flow(const DistributedGraph& graph, const std::span<const NodeIdentifier> sources,
                                            const std::span<const NodeIdentifier> sinks, const MaxFlowAlgorithm algorithm) {
    auto timer_setup = utility::RegionTimer{ "max-flow-setup" };

    utility::Exception::check(!sources.empty(), "MaximumFlow::compute_maximum_flow: there must be at least one source");
    utility::Exception::check(!sinks.empty(), "MaximumFlow::compute_maximum_flow: there must be at least one sink");

    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = graph.get_number_local_nodes();

    const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);
    const auto prefix_distribution = utility::calculate_prefix_sum(std::span{ node_distribution });

    // Translated and validated on every rank, so an invalid call fails on every rank instead of only on root
    const auto source_ids = translate_to_global_ids(sources, node_distribution, prefix_distribution, "source");
    const auto sink_ids = translate_to_global_ids(sinks, node_distribution, prefix_distribution, "sink");

    const auto source_set = std::unordered_set<std::size_t>{ source_ids.begin(), source_ids.end() };
    for (const auto sink_id : sink_ids) {
        utility::Exception::check(!source_set.contains(sink_id), "MaximumFlow::compute_maximum_flow: the sources and the sinks must be disjoint");
    }

    timer_setup.stop();

    // The distributed variant needs no gathering and no broadcast: it works on the distributed
    // graph directly and its collective reductions return the flow on every rank
    if (algorithm == MaxFlowAlgorithm::DistributedPushRelabel) {
        auto timer_compute = utility::RegionTimer{ "max-flow-compute" };

        const auto flow = DistributedPushRelabelVariant::compute_maximum_flow(graph, source_ids, sink_ids, node_distribution, prefix_distribution);
        const auto result = utility::safe_cast<flow_type>(flow);

        timer_compute.stop();

        mpiPP::MPIRegionTimerReport::reduce_and_print("MaximumFlow");
        utility::RegionTimer::init();

        return result;
    }

    auto timer_gather_arcs = utility::RegionTimer{ "max-flow-gather-arcs" };

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

    auto timer_compute = utility::RegionTimer{ "max-flow-compute" };

    auto max_flow = flow_type{ 0 };

    if (mpiPP::MPIInfo::is_root_rank()) {
        const auto total_number_nodes = utility::safe_cast<std::size_t>(std::reduce(node_distribution.begin(), node_distribution.end(), global_node_id_type{ 0 }));

        // The two extra nodes are the super source and the super sink that join the sources and the sinks
        auto network = max_flow::FlowNetwork{ total_number_nodes + 2 };
        const auto super_source = total_number_nodes;
        const auto super_sink = total_number_nodes + 1;

        auto total_capacity = max_flow::FlowNetwork::capacity_type{ 0 };

        for (const auto& arcs_of_rank : gathered_arcs) {
            for (const auto& [source, target, weight] : arcs_of_rank) {
                // A flow is only defined for positive capacities, so the weights enter by absolute
                // value; self loops and zero-weight arcs can never carry flow and are dropped.
                const auto capacity = std::abs(utility::safe_cast<max_flow::FlowNetwork::capacity_type>(weight));

                if (source == target || capacity == 0) {
                    continue;
                }

                network.add_arc(utility::safe_cast<std::size_t>(source), utility::safe_cast<std::size_t>(target), capacity);
                total_capacity += capacity;
            }
        }

        // No flow through a single node can exceed the total capacity of all arcs, so the total
        // capacity acts as an unbounded capacity for the super arcs.
        for (const auto source_id : source_ids) {
            network.add_arc(super_source, source_id, total_capacity);
        }

        for (const auto sink_id : sink_ids) {
            network.add_arc(sink_id, super_sink, total_capacity);
        }

        const auto flow = [&network, super_source, super_sink, algorithm]() -> max_flow::FlowNetwork::capacity_type {
            switch (algorithm) {
            case MaxFlowAlgorithm::FordFulkerson:
                return FordFulkersonVariant::compute_maximum_flow(network, super_source, super_sink);
            case MaxFlowAlgorithm::EdmondsKarp:
                return EdmondsKarpVariant::compute_maximum_flow(network, super_source, super_sink);
            case MaxFlowAlgorithm::Dinic:
                return DinicVariant::compute_maximum_flow(network, super_source, super_sink);
            case MaxFlowAlgorithm::OrlinKingRaoTarjan:
                return OrlinKingRaoTarjanVariant::compute_maximum_flow(network, super_source, super_sink);
            default:
                utility::Exception::fail("MaximumFlow::compute_maximum_flow: the algorithm is unknown");
            }
        }();

        max_flow = utility::safe_cast<flow_type>(flow);
    }

    timer_compute.stop();

    auto timer_broadcast = utility::RegionTimer{ "max-flow-broadcast" };

    const auto result = mpiPP::MPIBroadcasts::broadcast(max_flow, mpiPP::MPIRank::root_rank());

    timer_broadcast.stop();

    mpiPP::MPIRegionTimerReport::reduce_and_print("MaximumFlow");
    utility::RegionTimer::init();

    return result;
}
