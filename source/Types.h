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

#include <cpp-utility/hash/pair.hpp>
#include <cpp-utility/hash/tuple.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using mpi_rank_type = int;

using arc_id_type = std::uint32_t;
using node_id_type = std::uint32_t;
using global_node_id_type = std::uint64_t;

using weight_type = std::int32_t;
using distance_type = std::uint64_t;
using inverse_distance_type = double;
using flow_type = std::uint64_t;

enum class DeltaSteppingEpochType : std::uint8_t {
    PushBruteForce,
    PushCheck,
    PushShortLong,
    PushInnerShort,
    PullModel,
};

// Selects which vertex queue implementation Dijkstra::compute_sssp uses
// (see the types in metrics/paths/DijkstraVariants).
// BucketQueue requires integer distances and is therefore not valid for the inverse metric.
enum class DijkstraQueueType : std::uint8_t {
    PriorityQueue,
    FibonacciHeap,
    BucketQueue,
};

// Selects which algorithm BetweennessCentrality::compute_average_betweenness_centrality uses
// (see the types in metrics/paths/BetweennessCentralityVariants).
// Both compute identical values, but Brandes runs in O(V * (A + V log V)) while PathEnumeration
// materializes every shortest path and can take exponential time and memory.
enum class BetweennessCentralityAlgorithm : std::uint8_t {
    Brandes,
    PathEnumeration,
};

// Selects which community-detection algorithm Modularity::compute_modularity uses to find the
// partition whose modularity is reported (see the types in metrics/community/ModularityVariants).
// All four are deterministic, but they optimize different objectives, so the reported modularity
// differs between them: Louvain and Leiden greedily maximize the modularity itself (Leiden
// additionally refines the communities before every aggregation, which keeps them internally
// connected), LabelPropagation only follows the local weight majority, and Infomap minimizes the
// description length of a random walk. The value is therefore usually highest for Louvain and Leiden.
enum class ModularityAlgorithm : std::uint8_t {
    Louvain,
    Leiden,
    LabelPropagation,
    Infomap,
};

// Selects which algorithm MaximumFlow::compute_maximum_flow uses (see the types in
// metrics/flow/MaxFlowVariants). All variants are deterministic and compute the same value.
// The first four gather the arcs on the root rank and compute the flow there, so they only
// differ in their running time; DistributedPushRelabel instead keeps the graph distributed
// and computes the flow with all ranks.
enum class MaxFlowAlgorithm : std::uint8_t {
    FordFulkerson,
    EdmondsKarp,
    Dinic,
    OrlinKingRaoTarjan,
    DistributedPushRelabel,
};

// Selects which algorithm NetworkMotifs::compute_network_triple_motifs uses (see the types in
// metrics/local_structure/NetworkMotifsVariants). All variants are deterministic and compute the same
// fractions. Questions asks the owner of a node whether two of its neighbors are connected and is the
// only one that needs no remote arc reads, while the four RMA variants answer that question themselves
// by reading the arcs of the other MPI ranks, which removes the collective rounds from the algorithm.
// They differ in who counts a triangle that spans three MPI ranks (Rma and RmaAccumulate give it to the
// owner of its smallest node, RmaFair lets the three owners take turns), in which of two remote
// adjacencies is downloaded (RmaFewerArcs takes the smaller one), in whether the triples with both
// outer nodes on one other MPI rank are left to that rank (RmaFewerMessages), and in whether the counts
// are reduced collectively or accumulated one-sidedly (RmaAccumulate).
enum class NetworkMotifAlgorithm : std::uint8_t {
    Questions,
    Rma,
    RmaAccumulate,
    RmaFair,
    RmaFewerArcs,
    RmaFewerMessages,
};

// Selects which algorithm StronglyConnectedComponents::compute_strongly_connected_components uses
// (see the types in metrics/connectivity/SccVariants). All variants are deterministic and find the
// same components. TarjanGathered gathers the arcs on the root rank and runs the sequential algorithm
// there, while the other three keep the graph distributed: ForwardBackward peels one component per
// round, Coloring peels all components of a color layer per round, and Multistep combines trimming,
// one forward-backward round, and the coloring rounds.
enum class SccAlgorithm : std::uint8_t {
    TarjanGathered,
    ForwardBackward,
    Coloring,
    Multistep,
};

// Selects which AllPairsShortestPath method the CLI layer calls for the regular (weight-based) APSP;
// not a parameter of AllPairsShortestPath itself, which exposes compute_apsp and
// compute_apsp_delta_stepping as separate methods.
enum class SsspAlgorithm : std::uint8_t {
    Dijkstra,
    DeltaStepping,
};

// Below are more complex types which are declared here so that they are accessible in the MPIWrapper
struct AreaConnectivityInfo {
    mpi_rank_type source_rank{};
    node_id_type source_area_local_id{};
    mpi_rank_type target_rank{};
    node_id_type target_area_local_id{};
    weight_type weight{};
};

struct NodeIdentifier {
    mpi_rank_type owning_mpi_rank{};
    node_id_type node_id{};

    NodeIdentifier() = default;

    NodeIdentifier(const mpi_rank_type rank, const node_id_type node) noexcept {
        owning_mpi_rank = rank;
        node_id = node;
    }

    NodeIdentifier(const std::pair<mpi_rank_type, node_id_type>& pair) noexcept {
        owning_mpi_rank = pair.first;
        node_id = pair.second;
    }

    NodeIdentifier(const std::tuple<mpi_rank_type, node_id_type>& tuple) noexcept {
        owning_mpi_rank = std::get<0>(tuple);
        node_id = std::get<1>(tuple);
    }

    [[nodiscard]] bool has_uninitialized_rank() const noexcept {
        return owning_mpi_rank == std::numeric_limits<mpi_rank_type>::max();
    }

    [[nodiscard]] bool operator==(const NodeIdentifier& other) const noexcept = default;

    [[nodiscard]] static NodeIdentifier uninitialized() noexcept {
        return NodeIdentifier{ std::numeric_limits<mpi_rank_type>::max(), std::numeric_limits<node_id_type>::max() };
    }

    [[nodiscard]] operator std::pair<mpi_rank_type, node_id_type>() const noexcept {
        return { owning_mpi_rank, node_id };
    }

    [[nodiscard]] operator std::tuple<mpi_rank_type, node_id_type>() const noexcept {
        return { owning_mpi_rank, node_id };
    }

    friend std::ostream& operator<<(std::ostream& out, const NodeIdentifier& ni) {
        out << '(' << ni.owning_mpi_rank << ", " << ni.node_id << ')';
        return out;
    }
};

namespace std {
template <>
struct hash<NodeIdentifier> {
    [[nodiscard]] std::size_t operator()(const NodeIdentifier& value) const {
        const auto tup = static_cast<std::tuple<mpi_rank_type, node_id_type>>(value);
        const auto hasher = utility::hash<std::tuple<mpi_rank_type, node_id_type>>{};
        return hasher(tup);
    }
};
} // namespace std

struct MinMax {
    arc_id_type min{ std::numeric_limits<arc_id_type>::max() };
    arc_id_type max{ std::numeric_limits<arc_id_type>::min() };
};

using AreaConnectivityMap = std::unordered_map<std::pair<std::string, std::string>, weight_type, utility::hash<std::pair<std::string, std::string>>>;

/**
 * @brief The strongly connected components of a graph, summarized by their sizes.
 *		See StronglyConnectedComponents::compute_strongly_connected_components.
 */
struct SccResult {
    /** The number of strongly connected components in the whole graph */
    global_node_id_type number_components{};

    /** The number of nodes of the largest strongly connected component */
    global_node_id_type largest_component_size{};

    /**
     * The size distribution of the components, i.e., <member>[i] == j indicates that j components
     * consist of exactly i nodes. Sizes without a component are absent. Note that this counts the
     * components of a size, unlike ApspGlobalResult::cluster_sizes, which counts their nodes.
     */
    std::unordered_map<global_node_id_type, global_node_id_type> component_sizes{};
};

/**
 * @brief The rich-club coefficients of a graph, see RichClub::compute_rich_club_coefficients.
 *		All three vectors have the same length and are indexed by the degree threshold, which runs from
 *		zero up to the largest one whose club still holds at least two nodes. They are empty if not
 *		even the threshold zero does, i.e., if fewer than two nodes have an arc at all.
 */
struct RichClubResult {
    /** The rich-club coefficient of the threshold, i.e., the arc density among its club, from [0, 1] */
    std::vector<double> coefficients{};

    /** How many nodes have a degree larger than the threshold, i.e., how large its club is */
    std::vector<global_node_id_type> club_sizes{};

    /** How many arcs run from one member of the club of the threshold to a different one */
    std::vector<distance_type> club_arc_counts{};
};

template <typename d_type = distance_type>
struct ApspGlobalResult {
    double average_shortest_path_length{};
    double average_efficiency{};
    d_type diameter{};
    global_node_id_type number_disconnected_pairs{};
    std::unordered_map<global_node_id_type, global_node_id_type> cluster_sizes{};
};
