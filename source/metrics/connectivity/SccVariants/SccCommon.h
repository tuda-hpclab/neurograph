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

#include "metrics/connectivity/SccVariants/DistributedSccState.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi-wrapper/collectives/MPIGatherV.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/reductions/MPIAdvancedReductions.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Shared building blocks of the strongly-connected-components variants (see the sibling headers
 *		in metrics/connectivity/SccVariants): the sequential algorithm that every variant falls back to
 *		once a part of the graph is small enough to fit on one rank, the gathering that feeds it, and
 *		the reduction that turns the local component sizes into the global result.
 */
class SccCommon {
public:
    /**
     * @brief Determines the strongly connected components of a graph on densely numbered nodes with
     *		Tarjan's algorithm (1972), "Depth-first search and linear graph algorithms". The recursion
     *		is spelled out as an explicit stack, so the depth of the depth-first search is bounded by
     *		memory instead of by the call stack.
     *		Purely local, performs no communication. Complexity is O(V + A).
     * @param number_nodes The number of nodes, which are numbered 0, ..., number_nodes - 1
     * @param arcs The arcs as pairs of (source, target); self arcs and duplicates are allowed
     * @exception Throws an Exception if an arc references a node that does not exist
     * @return One entry per component, holding the number of its nodes; the components appear in the
     *		order in which the depth-first search completes them
     */
    [[nodiscard]] static std::vector<std::size_t> compute_component_sizes(const std::size_t number_nodes, const std::span<const std::pair<std::size_t, std::size_t>> arcs) {
        // The arcs of a node have to be visited one after the other, which needs them grouped by source
        auto arc_begins = std::vector<std::size_t>(number_nodes + 1, std::size_t{ 0 });
        for (const auto& [source, target] : arcs) {
            utility::Exception::check(source < number_nodes && target < number_nodes, "SccCommon::compute_component_sizes: The arc {} -> {} references a node outside of the {} nodes", source, target, number_nodes);
            ++arc_begins[source + 1];
        }

        for (auto node = std::size_t{ 0 }; node < number_nodes; ++node) {
            arc_begins[node + 1] += arc_begins[node];
        }

        auto arc_cursors = arc_begins;
        auto arc_targets = std::vector<std::size_t>(arcs.size(), std::size_t{ 0 });
        for (const auto& [source, target] : arcs) {
            arc_targets[arc_cursors[source]] = target;
            ++arc_cursors[source];
        }

        // The depth-first index of a node, its lowest reachable index, and whether it waits for its
        // component on the component stack
        auto depth_first_indices = std::vector<std::size_t>(number_nodes, unvisited);
        auto lowest_indices = std::vector<std::size_t>(number_nodes, std::size_t{ 0 });
        auto on_component_stack = std::vector<std::uint8_t>(number_nodes, std::uint8_t{ 0 });

        auto component_stack = std::vector<std::size_t>{};
        auto call_stack = std::vector<SearchFrame>{};
        auto next_depth_first_index = std::size_t{ 0 };

        auto component_sizes = std::vector<std::size_t>{};

        const auto enter_node = [&](const std::size_t node) {
            depth_first_indices[node] = next_depth_first_index;
            lowest_indices[node] = next_depth_first_index;
            ++next_depth_first_index;

            on_component_stack[node] = 1;
            component_stack.emplace_back(node);
            call_stack.emplace_back(SearchFrame{ node, arc_begins[node] });
        };

        for (auto root = std::size_t{ 0 }; root < number_nodes; ++root) {
            if (depth_first_indices[root] != unvisited) {
                continue;
            }

            enter_node(root);

            while (!call_stack.empty()) {
                const auto node = call_stack.back().node;

                if (call_stack.back().next_arc < arc_begins[node + 1]) {
                    const auto arc_index = call_stack.back().next_arc;
                    // The cursor has to advance before entering a node, which invalidates the reference
                    ++call_stack.back().next_arc;

                    const auto target = arc_targets[arc_index];

                    if (depth_first_indices[target] == unvisited) {
                        enter_node(target);
                    } else if (on_component_stack[target] != 0) {
                        lowest_indices[node] = std::min(lowest_indices[node], depth_first_indices[target]);
                    }

                    continue;
                }

                // The node has no arcs left, so it is the root of its component iff nothing below it
                // reaches past it
                if (lowest_indices[node] == depth_first_indices[node]) {
                    auto number_nodes_in_component = std::size_t{ 0 };

                    while (true) {
                        const auto member = component_stack.back();
                        component_stack.pop_back();
                        on_component_stack[member] = 0;
                        ++number_nodes_in_component;

                        if (member == node) {
                            break;
                        }
                    }

                    component_sizes.emplace_back(number_nodes_in_component);
                }

                call_stack.pop_back();

                if (!call_stack.empty()) {
                    const auto parent = call_stack.back().node;
                    lowest_indices[parent] = std::min(lowest_indices[parent], lowest_indices[node]);
                }
            }
        }

        return component_sizes;
    }

    /**
     * @brief Gathers the unprocessed part of the graph on the root rank and determines its components
     *		there with the sequential algorithm. The ranks only know whether their own nodes are still
     *		active, so they contribute every arc that starts at an active node and the root rank drops
     *		the arcs whose target has already been processed.
     *		Must be called on every MPI rank; only the root rank returns the sizes, which is what
     *		all_assemble_result expects.
     * @param state The state whose active part is gathered
     * @return On the root rank <return>[i] == j indicates that j of the gathered components consist of
     *		i nodes; an empty map on every other rank
     */
    [[nodiscard]] static std::unordered_map<global_node_id_type, global_node_id_type> all_compute_gathered_component_sizes(const DistributedSccState& state) {
        auto local_global_ids = state.get_active_global_ids();
        auto local_arcs = state.get_arcs_from_active_nodes();

        const auto gathered_global_ids = mpiPP::MPICollectives::gatherv(std::move(local_global_ids), mpiPP::MPIRank::root_rank());
        const auto gathered_arcs = mpiPP::MPICollectives::gatherv(std::move(local_arcs), mpiPP::MPIRank::root_rank());

        auto size_histogram = std::unordered_map<global_node_id_type, global_node_id_type>{};

        if (!mpiPP::MPIInfo::is_root_rank()) {
            return size_histogram;
        }

        // Numbering the gathered nodes in the order of the ranks turns their global ids into the dense
        // node ids the sequential algorithm works on
        auto dense_ids = std::unordered_map<global_node_id_type, std::size_t>{};
        for (const auto& global_ids_of_rank : gathered_global_ids) {
            for (const auto global_id : global_ids_of_rank) {
                dense_ids.emplace(global_id, dense_ids.size());
            }
        }

        auto arcs = std::vector<std::pair<std::size_t, std::size_t>>{};
        for (const auto& arcs_of_rank : gathered_arcs) {
            for (const auto& [source_global_id, target_global_id] : arcs_of_rank) {
                const auto source = dense_ids.find(source_global_id);
                const auto target = dense_ids.find(target_global_id);

                if (source == dense_ids.end() || target == dense_ids.end()) {
                    continue;
                }

                arcs.emplace_back(source->second, target->second);
            }
        }

        for (const auto number_nodes_in_component : compute_component_sizes(dense_ids.size(), arcs)) {
            ++size_histogram[utility::safe_cast<global_node_id_type>(number_nodes_in_component)];
        }

        return size_histogram;
    }

    /**
     * @brief Combines the component sizes all ranks found into the global result. Every component must
     *		be counted by exactly one rank, which is what DistributedSccState::all_count_component_sizes
     *		and all_compute_gathered_component_sizes guarantee.
     *		Must be called on every MPI rank, returns the same result on every MPI rank.
     * @param local_size_histogram The local contribution, where <param>[i] == j indicates that j
     *		components counted by this rank consist of i nodes
     * @return The strongly connected components of the whole graph, summarized by their sizes
     */
    [[nodiscard]] static SccResult all_assemble_result(const std::unordered_map<global_node_id_type, global_node_id_type>& local_size_histogram) {
        auto result = SccResult{};

        result.component_sizes = mpiPP::MPIAdvancedReductions::reduce_map(local_size_histogram);

        for (const auto& [number_nodes_in_component, number_components] : result.component_sizes) {
            if (number_components == 0) {
                continue;
            }

            result.number_components += number_components;
            result.largest_component_size = std::max(result.largest_component_size, number_nodes_in_component);
        }

        return result;
    }

private:
    /** The depth-first index of a node the search has not entered yet. */
    static constexpr auto unvisited = std::numeric_limits<std::size_t>::max();

    /** One entry of the explicit call stack: the node and the arc of it that is visited next. */
    struct SearchFrame {
        std::size_t node{};
        std::size_t next_arc{};
    };
};
