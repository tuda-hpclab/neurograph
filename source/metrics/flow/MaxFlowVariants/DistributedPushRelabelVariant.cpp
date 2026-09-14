/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "DistributedPushRelabelVariant.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/flow/MaxFlowVariants/FlowNetwork.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi-wrapper/collectives/MPIAllToAllV.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace {
/**
 * @brief A residual arc stored on the rank that owns its tail. The residual reverse arc is a
 *		separate entry on the rank that owns the head, and the two stay consistent because every
 *		push shrinks the pushed arc locally and grows the reverse arc through a push message.
 */
struct ResidualEntry {
    global_node_id_type head_global_id{};
    mpi_rank_type head_rank{};
    max_flow::FlowNetwork::capacity_type capacity{};
    // Index into heights if the head is local, into ghost_heights otherwise
    std::size_t head_height_index{};
};

// (target global id, origin global id, amount): the origin pushed amount towards the target, so
// the target gains excess and the residual reverse arc (target -> origin) gains capacity
using PushMessage = std::tuple<global_node_id_type, global_node_id_type, max_flow::FlowNetwork::capacity_type>;

// (head global id, tail global id) of a graph arc, sent to the head's rank so that it creates the
// entry of the residual reverse arc
using ReverseArcAnnouncement = std::tuple<global_node_id_type, global_node_id_type>;

/**
 * @brief The distributed residual network together with the push-relabel state. Every rank stores
 *		the excesses, the heights, and the outgoing residual arcs of its own nodes; the heights of
 *		remote neighbors are mirrored as ghost values and refreshed after every pulse.
 */
struct DistributedResidualNetwork {
    std::size_t number_local_nodes{};
    std::size_t number_ranks{};
    mpi_rank_type my_rank{};
    global_node_id_type my_prefix{};
    std::uint64_t number_total_nodes{};

    // The residual arcs of local node i are entries[entry_offsets[i]], ..., entries[entry_offsets[i + 1] - 1],
    // sorted by head_global_id, with at most one entry per head
    std::vector<ResidualEntry> entries{};
    std::vector<std::size_t> entry_offsets{};

    // The distinct global ids of all remote heads, sorted, so they are also grouped by owning rank
    std::vector<global_node_id_type> ghost_ids{};
    std::vector<std::size_t> ghost_offsets_per_rank{};

    // boundary_node_ids[r] holds the sorted local nodes whose heights rank r mirrors; by
    // construction this matches exactly the ghost ids that rank r keeps for this rank
    std::vector<std::vector<std::size_t>> boundary_node_ids{};

    std::vector<std::uint64_t> heights{};
    std::vector<std::uint64_t> ghost_heights{};
    std::vector<max_flow::FlowNetwork::capacity_type> excesses{};
    std::vector<std::size_t> current_entry_indices{};
    std::vector<char> needs_relabel{};
    std::vector<char> is_source{};
    std::vector<char> is_sink{};

    /**
     * @brief Builds the distributed residual network from the local out arcs. Every arc announces
     *		itself to the rank of its head, which stores the residual reverse arc, so both directions
     *		exist without relying on the in arcs of the graph.
     * @param graph The graph
     * @param node_distribution The number of nodes on each MPI rank
     * @param prefix_distribution The exclusive prefix sum of node_distribution
     */
    void build(const DistributedGraph& graph, const std::span<const global_node_id_type> node_distribution,
               const std::span<const global_node_id_type> prefix_distribution) {
        my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        number_ranks = utility::safe_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
        my_prefix = prefix_distribution[utility::safe_cast<std::size_t>(my_rank)];
        number_local_nodes = utility::safe_cast<std::size_t>(graph.get_number_local_nodes());
        number_total_nodes = std::reduce(node_distribution.begin(), node_distribution.end(), std::uint64_t{ 0 });

        struct RawEntry {
            std::size_t node{};
            global_node_id_type head_global_id{};
            mpi_rank_type head_rank{};
            max_flow::FlowNetwork::capacity_type capacity{};
        };

        auto raw_entries = std::vector<RawEntry>{};
        raw_entries.reserve(2 * utility::safe_cast<std::size_t>(graph.get_number_local_out_arcs()));

        auto announcements = std::vector<std::vector<ReverseArcAnnouncement>>(number_ranks);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            const auto tail_global_id = my_prefix + node;

            for (const auto& [target_rank, target_id, weight] : graph.get_out_arcs(my_rank, utility::safe_cast<node_id_type>(node))) {
                // A flow is only defined for positive capacities, so the weights enter by absolute
                // value; self loops and zero-capacity arcs can never carry flow and are dropped.
                const auto capacity = std::abs(utility::safe_cast<max_flow::FlowNetwork::capacity_type>(weight));
                const auto head_global_id = prefix_distribution[utility::safe_cast<std::size_t>(target_rank)] + target_id;

                if (head_global_id == tail_global_id || capacity == 0) {
                    continue;
                }

                raw_entries.push_back(RawEntry{ node, head_global_id, target_rank, capacity });
                announcements[utility::safe_cast<std::size_t>(target_rank)].emplace_back(head_global_id, tail_global_id);
            }
        }

        const auto received_announcements = mpiPP::MPICollectives::all_to_all_v(announcements);

        for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
            for (const auto& [head_global_id, tail_global_id] : received_announcements[rank]) {
                const auto node = utility::safe_cast<std::size_t>(head_global_id - my_prefix);
                raw_entries.push_back(RawEntry{ node, tail_global_id, utility::safe_cast<mpi_rank_type>(rank), 0 });
            }
        }

        // Parallel arcs between the same two nodes are merged, so that the pair (tail, head)
        // identifies a residual arc uniquely; merging does not change the maximum flow
        std::sort(raw_entries.begin(), raw_entries.end(), [](const RawEntry& left, const RawEntry& right) {
            return std::tie(left.node, left.head_global_id) < std::tie(right.node, right.head_global_id);
        });

        auto number_entries_per_node = std::vector<std::size_t>(number_local_nodes, 0);
        auto last_node = std::numeric_limits<std::size_t>::max();

        entries.reserve(raw_entries.size());

        for (const auto& raw_entry : raw_entries) {
            if (raw_entry.node == last_node && entries.back().head_global_id == raw_entry.head_global_id) {
                entries.back().capacity += raw_entry.capacity;
                continue;
            }

            entries.push_back(ResidualEntry{ raw_entry.head_global_id, raw_entry.head_rank, raw_entry.capacity, 0 });
            ++number_entries_per_node[raw_entry.node];
            last_node = raw_entry.node;
        }

        entry_offsets.assign(number_local_nodes + 1, 0);
        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            entry_offsets[node + 1] = entry_offsets[node] + number_entries_per_node[node];
        }

        for (const auto& entry : entries) {
            if (entry.head_rank != my_rank) {
                ghost_ids.push_back(entry.head_global_id);
            }
        }

        std::sort(ghost_ids.begin(), ghost_ids.end());
        ghost_ids.erase(std::unique(ghost_ids.begin(), ghost_ids.end()), ghost_ids.end());

        // The ghost ids are sorted globally, so the ghosts of one rank form a contiguous slice
        ghost_offsets_per_rank.assign(number_ranks + 1, ghost_ids.size());
        for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
            const auto first = std::lower_bound(ghost_ids.begin(), ghost_ids.end(), prefix_distribution[rank]);
            ghost_offsets_per_rank[rank] = utility::safe_cast<std::size_t>(first - ghost_ids.begin());
        }

        for (auto& entry : entries) {
            if (entry.head_rank == my_rank) {
                entry.head_height_index = utility::safe_cast<std::size_t>(entry.head_global_id - my_prefix);
            } else {
                const auto position = std::lower_bound(ghost_ids.begin(), ghost_ids.end(), entry.head_global_id);
                entry.head_height_index = utility::safe_cast<std::size_t>(position - ghost_ids.begin());
            }
        }

        boundary_node_ids.assign(number_ranks, {});
        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            for (auto entry_id = entry_offsets[node]; entry_id < entry_offsets[node + 1]; ++entry_id) {
                if (entries[entry_id].head_rank != my_rank) {
                    boundary_node_ids[utility::safe_cast<std::size_t>(entries[entry_id].head_rank)].push_back(node);
                }
            }
        }

        // The nodes were visited in ascending order, so removing adjacent duplicates suffices
        for (auto& nodes_of_rank : boundary_node_ids) {
            nodes_of_rank.erase(std::unique(nodes_of_rank.begin(), nodes_of_rank.end()), nodes_of_rank.end());
        }

        heights.assign(number_local_nodes, 0);
        ghost_heights.assign(ghost_ids.size(), 0);
        excesses.assign(number_local_nodes, 0);
        current_entry_indices.assign(number_local_nodes, 0);
        needs_relabel.assign(number_local_nodes, 0);
    }

    /**
     * @brief Marks the local sources and sinks. The sources are pinned at height V, the sinks at
     *		height 0, and neither ever pushes or relabels.
     * @param source_global_ids The distinct global ids of the sources
     * @param sink_global_ids The distinct global ids of the sinks
     */
    void mark_terminals(const std::span<const std::size_t> source_global_ids, const std::span<const std::size_t> sink_global_ids) {
        is_source.assign(number_local_nodes, 0);
        is_sink.assign(number_local_nodes, 0);

        for (const auto global_id : source_global_ids) {
            if (global_id >= my_prefix && global_id < my_prefix + number_local_nodes) {
                is_source[global_id - my_prefix] = 1;
                heights[global_id - my_prefix] = number_total_nodes;
            }
        }

        for (const auto global_id : sink_global_ids) {
            if (global_id >= my_prefix && global_id < my_prefix + number_local_nodes) {
                is_sink[global_id - my_prefix] = 1;
            }
        }
    }

    /**
     * @brief Returns the height of the head of the given entry, either from the local heights or
     *		from the mirrored ghost heights.
     * @param entry The residual arc
     * @return The height of the head
     */
    [[nodiscard]] std::uint64_t get_head_height(const ResidualEntry& entry) const {
        if (entry.head_rank == my_rank) {
            return heights[entry.head_height_index];
        }
        return ghost_heights[entry.head_height_index];
    }

    /**
     * @brief Finds the residual arc from the given local node towards the given head.
     * @param node The local node id of the tail
     * @param head_global_id The global id of the head
     * @return The index of the entry in entries
     */
    [[nodiscard]] std::size_t find_entry(const std::size_t node, const global_node_id_type head_global_id) const {
        const auto begin = entries.begin() + utility::safe_cast<std::ptrdiff_t>(entry_offsets[node]);
        const auto end = entries.begin() + utility::safe_cast<std::ptrdiff_t>(entry_offsets[node + 1]);

        const auto position = std::lower_bound(begin, end, head_global_id, [](const ResidualEntry& entry, const global_node_id_type value) {
            return entry.head_global_id < value;
        });

        utility::Exception::check(position != end && position->head_global_id == head_global_id,
                                  "DistributedPushRelabelVariant: node {} has no residual arc towards node {}", my_prefix + node, head_global_id);

        return utility::safe_cast<std::size_t>(position - entries.begin());
    }

    /**
     * @brief Saturates every arc that leaves a local source, which creates the initial excesses.
     * @return The resulting push messages, one vector per receiving rank
     */
    [[nodiscard]] std::vector<std::vector<PushMessage>> saturate_source_arcs() {
        auto push_messages = std::vector<std::vector<PushMessage>>(number_ranks);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_source[node] == 0) {
                continue;
            }

            for (auto entry_id = entry_offsets[node]; entry_id < entry_offsets[node + 1]; ++entry_id) {
                auto& entry = entries[entry_id];

                if (entry.capacity <= 0) {
                    continue;
                }

                push_messages[utility::safe_cast<std::size_t>(entry.head_rank)].emplace_back(entry.head_global_id, my_prefix + node, entry.capacity);
                entry.capacity = 0;
            }
        }

        return push_messages;
    }

    /**
     * @brief The push phase of one pulse: every node with excess pushes it over admissible arcs,
     *		i.e., residual arcs that lead exactly one height downwards judged by the heights of the
     *		previous pulse. A node that scans all its arcs without getting rid of its excess is
     *		marked for the relabel phase. The heights themselves stay untouched, so all pushes of
     *		one pulse are admissible with respect to the same consistent height snapshot.
     * @return The push messages of this pulse, one vector per receiving rank
     */
    [[nodiscard]] std::vector<std::vector<PushMessage>> push_excess() {
        auto push_messages = std::vector<std::vector<PushMessage>>(number_ranks);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_source[node] != 0 || is_sink[node] != 0) {
                continue;
            }

            while (excesses[node] > 0) {
                if (entry_offsets[node] + current_entry_indices[node] == entry_offsets[node + 1]) {
                    needs_relabel[node] = 1;
                    break;
                }

                auto& entry = entries[entry_offsets[node] + current_entry_indices[node]];

                if (entry.capacity > 0 && heights[node] == get_head_height(entry) + 1) {
                    const auto amount = std::min(excesses[node], entry.capacity);

                    entry.capacity -= amount;
                    excesses[node] -= amount;
                    push_messages[utility::safe_cast<std::size_t>(entry.head_rank)].emplace_back(entry.head_global_id, my_prefix + node, amount);
                } else {
                    ++current_entry_indices[node];
                }
            }
        }

        return push_messages;
    }

    /**
     * @brief Applies the exchanged push messages: the targets gain excess and the residual reverse
     *		arcs gain capacity. Additions commute, so the result does not depend on the message order.
     * @param push_messages_per_rank The received push messages, one vector per sending rank
     */
    void apply_push_messages(const std::vector<std::vector<PushMessage>>& push_messages_per_rank) {
        for (const auto& push_messages : push_messages_per_rank) {
            for (const auto& [target_global_id, origin_global_id, amount] : push_messages) {
                const auto node = utility::safe_cast<std::size_t>(target_global_id - my_prefix);

                excesses[node] += amount;
                entries[find_entry(node, origin_global_id)].capacity += amount;
            }
        }
    }

    /**
     * @brief The relabel phase of one pulse: every node that exhausted its arcs with leftover
     *		excess lifts itself to one above its lowest residual neighbor. Running after the push
     *		messages were applied is required for correctness: the relabel must also consider the
     *		residual reverse arcs that the incoming pushes of this pulse just created.
     */
    void relabel_exhausted_nodes() {
        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (needs_relabel[node] == 0) {
                continue;
            }

            needs_relabel[node] = 0;

            auto lowest_height = std::numeric_limits<std::uint64_t>::max();

            for (auto entry_id = entry_offsets[node]; entry_id < entry_offsets[node + 1]; ++entry_id) {
                if (entries[entry_id].capacity > 0) {
                    lowest_height = std::min(lowest_height, get_head_height(entries[entry_id]));
                }
            }

            // The excess arrived over some arc, so its residual reverse arc keeps the node connected
            utility::Exception::check(lowest_height != std::numeric_limits<std::uint64_t>::max(),
                                      "DistributedPushRelabelVariant: a node with excess has no residual arc");

            heights[node] = lowest_height + 1;
            current_entry_indices[node] = 0;
        }
    }

    /**
     * @brief Sends the values of all boundary nodes to the ranks that mirror them and refreshes
     *		the given ghost values with the received values.
     * @param local_values The per-node values of this rank, e.g., the heights
     * @param ghost_values The mirrored values of the remote neighbors to refresh
     */
    void refresh_ghost_values(const std::vector<std::uint64_t>& local_values, std::vector<std::uint64_t>& ghost_values) const {
        auto value_messages = std::vector<std::vector<std::uint64_t>>(number_ranks);

        for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
            value_messages[rank].reserve(boundary_node_ids[rank].size());

            for (const auto node : boundary_node_ids[rank]) {
                value_messages[rank].push_back(local_values[node]);
            }
        }

        const auto received_values = mpiPP::MPICollectives::all_to_all_v(value_messages);

        for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
            const auto offset = ghost_offsets_per_rank[rank];
            const auto number_expected = ghost_offsets_per_rank[rank + 1] - offset;

            utility::Exception::check(received_values[rank].size() == number_expected,
                                      "DistributedPushRelabelVariant: rank {} sent {} values, but {} ghost values are mirrored", rank,
                                      received_values[rank].size(), number_expected);

            std::copy(received_values[rank].begin(), received_values[rank].end(), ghost_values.begin() + utility::safe_cast<std::ptrdiff_t>(offset));
        }
    }

    /**
     * @brief Sends the heights of all boundary nodes to the ranks that mirror them and refreshes
     *		the local ghost heights with the received values.
     */
    void refresh_ghost_heights() {
        refresh_ghost_values(heights, ghost_heights);
    }

    /**
     * @brief Runs a distributed breadth-first search through the residual network towards the
     *		seeds: an eligible node is labeled with round k when it has a residual arc to a node
     *		that was labeled with k - 1, so every label is the length of a shortest residual path
     *		from the node to a seed. One round refreshes the ghost labels, labels the local nodes,
     *		and stops once no rank labeled anything.
     * @param levels In: 0 for the seeds and unreached everywhere else; out: the residual distances
     * @param ghost_levels The mirrored levels of the remote neighbors, initially unreached
     * @param eligible The local nodes that may be labeled; the ineligible ones block the search
     */
    void run_residual_bfs(std::vector<std::uint64_t>& levels, std::vector<std::uint64_t>& ghost_levels, const std::vector<char>& eligible) const {
        constexpr auto unreached = std::numeric_limits<std::uint64_t>::max();

        for (auto round = std::uint64_t{ 1 };; ++round) {
            refresh_ghost_values(levels, ghost_levels);

            auto labeled_locally = false;

            for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
                if (eligible[node] == 0 || levels[node] != unreached) {
                    continue;
                }

                for (auto entry_id = entry_offsets[node]; entry_id < entry_offsets[node + 1]; ++entry_id) {
                    const auto& entry = entries[entry_id];

                    if (entry.capacity <= 0) {
                        continue;
                    }

                    const auto head_level = entry.head_rank == my_rank ? levels[entry.head_height_index] : ghost_levels[entry.head_height_index];

                    if (head_level == round - 1) {
                        levels[node] = round;
                        labeled_locally = true;
                        break;
                    }
                }
            }

            if (!mpiPP::MPIReductions::all_reduce_or(labeled_locally)) {
                return;
            }
        }
    }

    /**
     * @brief Recomputes every height as an exact residual distance, the classic global-relabeling
     *		heuristic: a node that can still reach a sink in the residual network gets the length
     *		of a shortest such path, a node that cannot gets V plus its distance to the nearest
     *		source (its excess can only flow back), and the isolated rest, which can never carry
     *		excess, gets 2V. Exact distances are valid labels and never lie below the current
     *		heights, so correctness is unaffected, but the excess then takes shortest residual
     *		paths, which reduces the number of pulses substantially.
     */
    void relabel_globally() {
        constexpr auto unreached = std::numeric_limits<std::uint64_t>::max();

        auto sink_levels = std::vector<std::uint64_t>(number_local_nodes, unreached);
        auto ghost_levels = std::vector<std::uint64_t>(ghost_ids.size(), unreached);
        auto eligible = std::vector<char>(number_local_nodes, 0);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_sink[node] != 0) {
                sink_levels[node] = 0;
            }
            eligible[node] = is_source[node] == 0 && is_sink[node] == 0 ? 1 : 0;
        }

        run_residual_bfs(sink_levels, ghost_levels, eligible);

        auto source_levels = std::vector<std::uint64_t>(number_local_nodes, unreached);
        ghost_levels.assign(ghost_ids.size(), unreached);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_source[node] != 0) {
                source_levels[node] = 0;
            }
            // A node that reaches a sink keeps its sink distance; its neighbors reach the sink
            // through it, so it can never lie on a shortest path of the second search either
            if (sink_levels[node] != unreached) {
                eligible[node] = 0;
            }
        }

        run_residual_bfs(source_levels, ghost_levels, eligible);

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            // The sources stay pinned at V and the sinks at 0
            if (is_source[node] != 0 || is_sink[node] != 0) {
                continue;
            }

            if (sink_levels[node] != unreached) {
                heights[node] = sink_levels[node];
            } else if (source_levels[node] != unreached) {
                heights[node] = number_total_nodes + source_levels[node];
            } else {
                heights[node] = 2 * number_total_nodes;
            }

            // The heights may have risen, so previously scanned arcs can be admissible again
            current_entry_indices[node] = 0;
        }

        refresh_ghost_heights();
    }

    /**
     * @brief Counts the active nodes across all ranks, i.e., the nodes with positive excess that
     *		are neither a source nor a sink.
     * @return The global number of active nodes; the same on every rank
     */
    [[nodiscard]] std::uint64_t count_active_nodes_globally() const {
        auto number_local_active = std::uint64_t{ 0 };

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_source[node] == 0 && is_sink[node] == 0 && excesses[node] > 0) {
                ++number_local_active;
            }
        }

        return mpiPP::MPIReductions::all_reduce_sum(number_local_active);
    }

    /**
     * @brief Sums the excesses of all sinks across all ranks, which is the value of the flow.
     * @return The global excess of the sinks; the same on every rank
     */
    [[nodiscard]] max_flow::FlowNetwork::capacity_type compute_global_sink_excess() const {
        auto local_sink_excess = max_flow::FlowNetwork::capacity_type{ 0 };

        for (auto node = std::size_t{ 0 }; node < number_local_nodes; ++node) {
            if (is_sink[node] != 0) {
                local_sink_excess += excesses[node];
            }
        }

        return mpiPP::MPIReductions::all_reduce_sum(local_sink_excess);
    }
};
} // namespace

max_flow::FlowNetwork::capacity_type DistributedPushRelabelVariant::compute_maximum_flow(
    const DistributedGraph& graph, const std::span<const std::size_t> source_global_ids, const std::span<const std::size_t> sink_global_ids,
    const std::span<const global_node_id_type> node_distribution, const std::span<const global_node_id_type> prefix_distribution) {
    auto network = DistributedResidualNetwork{};
    network.build(graph, node_distribution, prefix_distribution);
    network.mark_terminals(source_global_ids, sink_global_ids);

    const auto initial_messages = network.saturate_source_arcs();
    network.apply_push_messages(mpiPP::MPICollectives::all_to_all_v(initial_messages));
    network.refresh_ghost_heights();

    // The synchronous algorithm needs O(V^2) pulses. The far larger limit only turns a potential
    // implementation error into a loud failure instead of a silent endless loop; it is clamped so
    // that the squaring cannot overflow.
    const auto clamped_number_nodes = std::min(network.number_total_nodes, std::uint64_t{ 1 } << 28);
    const auto pulse_limit = 16 * (clamped_number_nodes + 2) * (clamped_number_nodes + 2) + 64;

    // How many pulses lie between two global relabelings. The interval balances the cost of the
    // breadth-first searches against the pulses they save; correctness holds for any interval.
    constexpr auto global_relabel_interval = std::uint64_t{ 8 };

    auto pulse = std::uint64_t{ 0 };

    while (network.count_active_nodes_globally() > 0) {
        utility::Exception::check(pulse < pulse_limit, "DistributedPushRelabelVariant::compute_maximum_flow: the pulse limit {} was exceeded", pulse_limit);

        if (pulse % global_relabel_interval == 0) {
            network.relabel_globally();
        }

        ++pulse;

        const auto push_messages = network.push_excess();
        network.apply_push_messages(mpiPP::MPICollectives::all_to_all_v(push_messages));
        network.relabel_exhausted_nodes();
        network.refresh_ghost_heights();
    }

    return network.compute_global_sink_excess();
}
