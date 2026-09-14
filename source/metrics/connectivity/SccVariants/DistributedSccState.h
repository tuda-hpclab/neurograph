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

#include "graph/Arc.h"
#include "graph/DistributedGraph.h"
#include "metrics/counting/NodeDistributionCounter.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/prefix_sum.hpp>

#include <mpi-wrapper/collectives/MPIAllGather.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/comm_patterns_2/QuestionExchange.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief The working state that the strongly-connected-components variants share
 *		(see the sibling headers in metrics/connectivity/SccVariants).
 *
 *		Every MPI rank keeps one entry per local node: whether the node still belongs to the
 *		unprocessed part of the graph and, once its component is known, the label of that component.
 *		A label is the global id of one node of the component, so counting the nodes per label yields
 *		the component sizes. Which node of a component provides the label differs between the phases
 *		and does not matter for the result, which only depends on the partition itself.
 *
 *		All phases are frontier traversals in the shape of the delta-stepping epochs (see
 *		metrics/paths/DeltaSteppingVariants): a rank only ever reads the arcs of its own nodes and
 *		pushes the resulting updates to the ranks that own the other endpoint, which apply them to
 *		their own nodes. No phase reads remote state, so no RMA window is needed, and no phase touches
 *		remote arcs, so the arc caches of the graph stay empty. Since every state change is monotone,
 *		the receiving rank can drop duplicate updates locally.
 *
 *		Self arcs are ignored entirely: they connect a node to itself and can therefore neither put it
 *		into a component with another node nor keep it from being trimmed. The weights play no role.
 *
 *		Every method whose name starts with all_ performs collective communication and must be called
 *		on every MPI rank; the other ones are local. Deterministic: the pivot selection breaks ties by
 *		the smallest global id, the labels are extreme global ids, and everything else is set-based, so
 *		the components depend neither on the number of ranks nor on the order in which messages arrive.
 */
class DistributedSccState {
public:
    /** One arc between two global node ids, used when a part of the graph is gathered on one rank. */
    using GlobalArc = std::pair<global_node_id_type, global_node_id_type>;

    /** Whether a traversal follows the arcs in their own direction or against it. */
    enum class TraversalDirection : std::uint8_t {
        Forward,
        Backward,
    };

    /**
     * @brief Initializes the state such that every node still belongs to the unprocessed part of the
     *		graph. Determines the node distribution and the degrees without the self arcs.
     *		Performs collective communication and must therefore be called on every MPI rank.
     * @param distributed_graph The graph, must outlive the state
     */
    explicit DistributedSccState(const DistributedGraph& distributed_graph)
        : graph(distributed_graph)
        , my_rank(mpiPP::MPIInfo::get_my_rank().get_rank())
        , number_local_nodes(distributed_graph.get_number_local_nodes()) {
        const auto node_distribution = NodeDistributionCounter::all_count_node_distribution_global(graph);

        prefix_distribution = utility::calculate_prefix_sum(std::span<const global_node_id_type>{ node_distribution });
        total_number_nodes = std::reduce(node_distribution.begin(), node_distribution.end(), global_node_id_type{ 0 });

        assigned.resize(number_local_nodes, std::uint8_t{ 0 });
        labels.resize(number_local_nodes, global_node_id_type{ 0 });
        remaining_in_degrees.resize(number_local_nodes, arc_id_type{ 0 });
        remaining_out_degrees.resize(number_local_nodes, arc_id_type{ 0 });

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            auto in_degree = arc_id_type{ 0 };
            for (const auto& in_arc : graph.get_in_arcs(my_rank, node_id)) {
                if (!is_self_arc(in_arc.source_rank, in_arc.source_id, node_id)) {
                    ++in_degree;
                }
            }

            auto out_degree = arc_id_type{ 0 };
            for (const auto& out_arc : graph.get_out_arcs(my_rank, node_id)) {
                if (!is_self_arc(out_arc.target_rank, out_arc.target_id, node_id)) {
                    ++out_degree;
                }
            }

            remaining_in_degrees[node_id] = in_degree;
            remaining_out_degrees[node_id] = out_degree;
        }
    }

    /**
     * @brief Returns the number of nodes on the current MPI rank
     * @return The number of local nodes
     */
    [[nodiscard]] node_id_type get_number_local_nodes() const noexcept {
        return number_local_nodes;
    }

    /**
     * @brief Returns the global id of a local node, i.e., its id within the concatenation of all ranks
     * @param node_id The local node id
     * @return The global node id
     */
    [[nodiscard]] global_node_id_type get_global_id(const node_id_type node_id) const {
        return prefix_distribution[utility::safe_cast<std::size_t>(my_rank)] + node_id;
    }

    /**
     * @brief Returns the node a global id refers to, i.e., the inverse of get_global_id
     * @param global_id The global node id, must exist
     * @exception Throws an Exception if no node has that global id
     * @return The owning MPI rank and the node id on that rank
     */
    [[nodiscard]] NodeIdentifier get_node_identifier(const global_node_id_type global_id) const {
        utility::Exception::check(global_id < total_number_nodes, "DistributedSccState::get_node_identifier: The global id {} does not exist, the graph has {} nodes", global_id, total_number_nodes);

        // The prefix of a rank without nodes repeats the prefix of its predecessor, so the upper bound,
        // which skips those repetitions, locates the rank that really owns the node
        const auto first_larger_prefix = std::ranges::upper_bound(prefix_distribution, global_id);
        const auto rank_index = utility::safe_cast<std::size_t>(first_larger_prefix - prefix_distribution.begin()) - 1;

        return NodeIdentifier{ utility::safe_cast<mpi_rank_type>(rank_index), utility::safe_cast<node_id_type>(global_id - prefix_distribution[rank_index]) };
    }

    /**
     * @brief Returns whether a local node still belongs to the unprocessed part of the graph
     * @param node_id The local node id
     * @return true iff the component of the node is not known yet
     */
    [[nodiscard]] bool is_active(const node_id_type node_id) const {
        return assigned[node_id] == 0;
    }

    /**
     * @brief Counts the nodes that still belong to the unprocessed part of the graph.
     *		Must be called on every MPI rank, returns the result on every MPI rank.
     * @return The number of active nodes in the whole graph
     */
    [[nodiscard]] global_node_id_type all_count_active_nodes() const {
        auto local_number_active_nodes = global_node_id_type{ 0 };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id)) {
                ++local_number_active_nodes;
            }
        }

        return mpiPP::MPIReductions::all_reduce_sum(local_number_active_nodes);
    }

    /**
     * @brief Removes every node that cannot lie on a cycle because it has no in arc or no out arc left
     *		within the unprocessed part of the graph, and gives it a component of its own. Removing such
     *		a node can expose further ones, so this repeats until nothing changes anymore, which takes
     *		as many rounds as the removed part is deep.
     *		Must be called on every MPI rank.
     */
    void all_trim() {
        while (true) {
            auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<std::uint8_t, node_id_type>(number_local_nodes);
            auto removed_a_node = false;

            for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
                if (!is_active(node_id) || (remaining_in_degrees[node_id] != 0 && remaining_out_degrees[node_id] != 0)) {
                    continue;
                }

                // Every cycle through the node would have to enter it and to leave it again, so the
                // node is a component of its own
                assign(node_id, get_global_id(node_id));
                removed_a_node = true;

                const auto& in_arcs = graph.get_in_arcs(my_rank, node_id);
                const auto& out_arcs = graph.get_out_arcs(my_rank, node_id);

                auto questions = question_list_type<std::uint8_t>{};
                questions.reserve(in_arcs.size() + out_arcs.size());

                for (const auto& out_arc : out_arcs) {
                    if (!is_self_arc(out_arc.target_rank, out_arc.target_id, node_id)) {
                        questions.emplace_back(out_arc.target_rank, out_arc.target_id, in_degree_decrement);
                    }
                }

                for (const auto& in_arc : in_arcs) {
                    if (!is_self_arc(in_arc.source_rank, in_arc.source_id, node_id)) {
                        questions.emplace_back(in_arc.source_rank, in_arc.source_id, out_degree_decrement);
                    }
                }

                question_builder.add_questions(node_id, std::move(questions));
            }

            // All ranks decide together, so the ranks that removed nothing still take part in the exchange
            if (mpiPP::MPIReductions::all_reduce_and(!removed_a_node)) {
                return;
            }

            const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());

            for (const auto& question : questions_to_answer) {
                const auto node_id = question.target_node;

                if (!is_active(node_id)) {
                    continue;
                }

                // Each arc is announced exactly once, namely when its other endpoint is removed, so the
                // degrees stay exact; the guards only document that they cannot underflow
                if (question.parameter == in_degree_decrement && remaining_in_degrees[node_id] > 0) {
                    --remaining_in_degrees[node_id];
                } else if (question.parameter == out_degree_decrement && remaining_out_degrees[node_id] > 0) {
                    --remaining_out_degrees[node_id];
                }
            }
        }
    }

    /**
     * @brief Selects the node the next forward-backward round starts from. The classic heuristic is the
     *		largest product of the degrees within the unprocessed part of the graph: a node whose product
     *		is zero cannot lie on a cycle at all, while a node with many arcs on both sides is likely to
     *		lie in a large component. The degrees are the ones all_trim left behind and are not updated
     *		afterwards, which only affects how good the heuristic is, not the result.
     *		Must be called on every MPI rank, returns the same node on every MPI rank.
     * @return The pivot node, or an uninitialized identifier if no rank has an active node left
     */
    [[nodiscard]] NodeIdentifier all_select_pivot() const {
        // A candidate is its degree product together with its global id; a rank without an active node
        // reports the largest global id, which no node can have and which every real candidate beats
        auto local_candidate = std::pair{ global_node_id_type{ 0 }, std::numeric_limits<global_node_id_type>::max() };

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (!is_active(node_id)) {
                continue;
            }

            const auto in_degree = utility::safe_cast<global_node_id_type>(remaining_in_degrees[node_id]);
            const auto out_degree = utility::safe_cast<global_node_id_type>(remaining_out_degrees[node_id]);
            const auto candidate = std::pair{ in_degree * out_degree, get_global_id(node_id) };

            if (is_better_candidate(candidate, local_candidate)) {
                local_candidate = candidate;
            }
        }

        const auto candidates = mpiPP::MPICollectives::all_gather(local_candidate);

        auto best_candidate = local_candidate;
        for (const auto& candidate : candidates) {
            if (is_better_candidate(candidate, best_candidate)) {
                best_candidate = candidate;
            }
        }

        if (best_candidate.second == std::numeric_limits<global_node_id_type>::max()) {
            return NodeIdentifier::uninitialized();
        }

        return get_node_identifier(best_candidate.second);
    }

    /**
     * @brief Determines the component of the pivot and removes it from the unprocessed part of the
     *		graph: the component consists of the nodes the pivot reaches and that reach the pivot, so
     *		one forward and one backward traversal suffice (Fleischer, Hendrickson, and Pinar (2000),
     *		"On identifying strongly connected components in parallel"). Removing a whole component
     *		leaves the other components untouched, because a cycle that leaves a component can never
     *		return to it.
     *		Must be called on every MPI rank with the same pivot.
     * @param pivot The node whose component is determined, must still be active
     */
    void all_assign_component_of(const NodeIdentifier pivot) {
        const auto reached_forward = all_traverse_from(pivot, TraversalDirection::Forward);
        const auto reached_backward = all_traverse_from(pivot, TraversalDirection::Backward);

        const auto is_in_component = [&reached_forward, &reached_backward, this](const node_id_type node_id) {
            return is_active(node_id) && reached_forward[node_id] != 0 && reached_backward[node_id] != 0;
        };

        // The pivot belongs to its own component, so the component is never empty
        auto local_label = std::numeric_limits<global_node_id_type>::max();
        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_in_component(node_id)) {
                local_label = std::min(local_label, get_global_id(node_id));
            }
        }

        const auto label = mpiPP::MPIReductions::all_reduce_min(local_label);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_in_component(node_id)) {
                assign(node_id, label);
            }
        }
    }

    /**
     * @brief Determines the components of one color layer and removes them from the unprocessed part of
     *		the graph (Orzan (2004), "On Distributed Verification and Verified Distribution"): every
     *		active node starts with its own global id as its color, the largest color travels along the
     *		arcs until nothing changes anymore, and a node that kept its own color is the root of its
     *		color. Since all nodes of a component share their color, the component of a root consists of
     *		the nodes of its color that reach it, which one backward traversal from all roots at once
     *		collects. At least the component of the largest active global id is found, so repeated rounds
     *		terminate.
     *		Must be called on every MPI rank.
     */
    void all_assign_components_by_coloring() {
        auto colors = std::vector<global_node_id_type>(number_local_nodes, global_node_id_type{ 0 });
        auto queued = std::vector<std::uint8_t>(number_local_nodes, std::uint8_t{ 0 });
        auto frontier = std::vector<node_id_type>{};

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id)) {
                colors[node_id] = get_global_id(node_id);
                frontier.emplace_back(node_id);
            }
        }

        // The largest color of all nodes that reach a node travels forward until it is stable
        while (true) {
            if (mpiPP::MPIReductions::all_reduce_and(frontier.empty())) {
                break;
            }

            auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<global_node_id_type, node_id_type>(number_local_nodes);

            for (const auto node_id : frontier) {
                const auto& out_arcs = graph.get_out_arcs(my_rank, node_id);

                auto questions = question_list_type<global_node_id_type>{};
                questions.reserve(out_arcs.size());

                for (const auto& out_arc : out_arcs) {
                    questions.emplace_back(out_arc.target_rank, out_arc.target_id, colors[node_id]);
                }

                question_builder.add_questions(node_id, std::move(questions));
            }

            const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());

            for (const auto node_id : frontier) {
                queued[node_id] = 0;
            }
            frontier.clear();

            for (const auto& question : questions_to_answer) {
                const auto node_id = question.target_node;

                if (!is_active(node_id) || question.parameter <= colors[node_id]) {
                    continue;
                }

                colors[node_id] = question.parameter;

                // A node can be improved several times within one exchange, but it may enter the
                // frontier only once: a questioner may add its questions in a single call only
                if (queued[node_id] == 0) {
                    queued[node_id] = 1;
                    frontier.emplace_back(node_id);
                }
            }
        }

        auto in_component = std::vector<std::uint8_t>(number_local_nodes, std::uint8_t{ 0 });

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id) && colors[node_id] == get_global_id(node_id)) {
                in_component[node_id] = 1;
                frontier.emplace_back(node_id);
            }
        }

        // Every root collects its component by travelling backwards within its own color
        while (true) {
            if (mpiPP::MPIReductions::all_reduce_and(frontier.empty())) {
                break;
            }

            auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<global_node_id_type, node_id_type>(number_local_nodes);

            for (const auto node_id : frontier) {
                const auto& in_arcs = graph.get_in_arcs(my_rank, node_id);

                auto questions = question_list_type<global_node_id_type>{};
                questions.reserve(in_arcs.size());

                for (const auto& in_arc : in_arcs) {
                    questions.emplace_back(in_arc.source_rank, in_arc.source_id, colors[node_id]);
                }

                question_builder.add_questions(node_id, std::move(questions));
            }

            const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());

            frontier.clear();

            for (const auto& question : questions_to_answer) {
                const auto node_id = question.target_node;

                // Belonging to the component is monotone, so the receiver drops the duplicates that
                // several arcs into the same node cause
                if (!is_active(node_id) || in_component[node_id] != 0 || colors[node_id] != question.parameter) {
                    continue;
                }

                in_component[node_id] = 1;
                frontier.emplace_back(node_id);
            }
        }

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id) && in_component[node_id] != 0) {
                assign(node_id, colors[node_id]);
            }
        }
    }

    /**
     * @brief Counts the nodes of every component whose label refers to a local node: each node that
     *		knows its component reports itself to the rank that owns its label, which then knows the
     *		sizes of the components it represents.
     *		Must be called on every MPI rank; the returned counts of all ranks together cover every
     *		component of the graph exactly once. Nodes that are still active are not counted.
     * @return <return>[i] == j indicates that j components represented by this rank consist of i nodes
     */
    [[nodiscard]] std::unordered_map<global_node_id_type, global_node_id_type> all_count_component_sizes() const {
        auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<std::uint8_t, node_id_type>(number_local_nodes);

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id)) {
                continue;
            }

            const auto label_node = get_node_identifier(labels[node_id]);

            auto questions = question_list_type<std::uint8_t>{};
            questions.emplace_back(label_node.owning_mpi_rank, label_node.node_id, node_of_component);

            question_builder.add_questions(node_id, std::move(questions));
        }

        const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());

        auto number_nodes_per_component = std::vector<global_node_id_type>(number_local_nodes, global_node_id_type{ 0 });
        for (const auto& question : questions_to_answer) {
            ++number_nodes_per_component[question.target_node];
        }

        auto size_histogram = std::unordered_map<global_node_id_type, global_node_id_type>{};
        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (number_nodes_per_component[node_id] > 0) {
                ++size_histogram[number_nodes_per_component[node_id]];
            }
        }

        return size_histogram;
    }

    /**
     * @brief Returns the global ids of the local nodes that still belong to the unprocessed part of the
     *		graph, in increasing order
     * @return The global ids of the active local nodes
     */
    [[nodiscard]] std::vector<global_node_id_type> get_active_global_ids() const {
        auto active_global_ids = std::vector<global_node_id_type>{};

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (is_active(node_id)) {
                active_global_ids.emplace_back(get_global_id(node_id));
            }
        }

        return active_global_ids;
    }

    /**
     * @brief Returns the arcs that start at an active local node, without the self arcs. Whether the
     *		target is active as well is not known locally and has to be checked by the receiver against
     *		the gathered ids.
     * @return The arcs as pairs of global node ids
     */
    [[nodiscard]] std::vector<GlobalArc> get_arcs_from_active_nodes() const {
        auto arcs = std::vector<GlobalArc>{};

        for (auto node_id = node_id_type{ 0 }; node_id < number_local_nodes; ++node_id) {
            if (!is_active(node_id)) {
                continue;
            }

            const auto source_global_id = get_global_id(node_id);

            for (const auto& out_arc : graph.get_out_arcs(my_rank, node_id)) {
                if (is_self_arc(out_arc.target_rank, out_arc.target_id, node_id)) {
                    continue;
                }

                const auto target_global_id = prefix_distribution[utility::safe_cast<std::size_t>(out_arc.target_rank)] + out_arc.target_id;
                arcs.emplace_back(source_global_id, target_global_id);
            }
        }

        return arcs;
    }

private:
    template <typename QuestionParameter>
    using question_list_type = std::vector<mpiPP::comm_patterns_2::TargetedQuestion<QuestionParameter, node_id_type>>;

    /** Announces to an out neighbor of a removed node that its in degree shrinks. */
    static constexpr auto in_degree_decrement = std::uint8_t{ 0 };

    /** Announces to an in neighbor of a removed node that its out degree shrinks. */
    static constexpr auto out_degree_decrement = std::uint8_t{ 1 };

    /** Announces to the rank that owns a label that one more node belongs to that component. */
    static constexpr auto node_of_component = std::uint8_t{ 0 };

    /** Announces to a neighbor of a frontier node that the traversal reached it. */
    static constexpr auto traversal_step = std::uint8_t{ 0 };

    /**
     * @brief Compares two pivot candidates: the larger degree product wins, ties are broken by the
     *		smaller global id, which keeps the choice independent of the number of ranks
     * @param candidate The candidate to check
     * @param other The candidate to compare against
     * @return true iff candidate is the better pivot
     */
    [[nodiscard]] static bool is_better_candidate(const std::pair<global_node_id_type, global_node_id_type>& candidate,
                                                  const std::pair<global_node_id_type, global_node_id_type>& other) {
        if (candidate.first != other.first) {
            return candidate.first > other.first;
        }

        return candidate.second < other.second;
    }

    /**
     * @brief Returns whether an arc of a local node connects that node to itself
     * @param other_rank The rank of the other endpoint
     * @param other_id The node id of the other endpoint
     * @param node_id The local node the arc belongs to
     * @return true iff the arc is a self arc
     */
    [[nodiscard]] bool is_self_arc(const mpi_rank_type other_rank, const node_id_type other_id, const node_id_type node_id) const {
        return other_rank == my_rank && other_id == node_id;
    }

    /**
     * @brief Removes a node from the unprocessed part of the graph and stores the label of its component
     * @param node_id The local node id
     * @param label The global id of a node of its component
     */
    void assign(const node_id_type node_id, const global_node_id_type label) {
        assigned[node_id] = 1;
        labels[node_id] = label;
    }

    /**
     * @brief Traverses the active part of the graph from one node, following the arcs either in their
     *		own direction or against it. One round of the loop advances the traversal by one arc, so it
     *		takes as many collective exchanges as the reached part is deep.
     *		Must be called on every MPI rank with the same source and direction.
     * @param source The node to start at; a node that is not active reaches nothing
     * @param direction Whether to follow the out arcs or the in arcs
     * @return <return>[node_id] != 0 indicates that the local node is active and was reached
     */
    [[nodiscard]] std::vector<std::uint8_t> all_traverse_from(const NodeIdentifier source, const TraversalDirection direction) const {
        auto reached = std::vector<std::uint8_t>(number_local_nodes, std::uint8_t{ 0 });
        auto frontier = std::vector<node_id_type>{};

        if (my_rank == source.owning_mpi_rank && is_active(source.node_id)) {
            reached[source.node_id] = 1;
            frontier.emplace_back(source.node_id);
        }

        while (true) {
            if (mpiPP::MPIReductions::all_reduce_and(frontier.empty())) {
                break;
            }

            auto question_builder = mpiPP::comm_patterns_2::QuestionBuilder<std::uint8_t, node_id_type>(number_local_nodes);

            for (const auto node_id : frontier) {
                auto questions = question_list_type<std::uint8_t>{};

                if (direction == TraversalDirection::Forward) {
                    const auto& out_arcs = graph.get_out_arcs(my_rank, node_id);
                    questions.reserve(out_arcs.size());

                    for (const auto& out_arc : out_arcs) {
                        questions.emplace_back(out_arc.target_rank, out_arc.target_id, traversal_step);
                    }
                } else {
                    const auto& in_arcs = graph.get_in_arcs(my_rank, node_id);
                    questions.reserve(in_arcs.size());

                    for (const auto& in_arc : in_arcs) {
                        questions.emplace_back(in_arc.source_rank, in_arc.source_id, traversal_step);
                    }
                }

                question_builder.add_questions(node_id, std::move(questions));
            }

            const auto questions_to_answer = mpiPP::comm_patterns_2::exchange_questions(question_builder.finalize());

            frontier.clear();

            for (const auto& question : questions_to_answer) {
                const auto node_id = question.target_node;

                // Being reached is monotone, so the receiver drops the duplicates that several arcs
                // into the same node cause
                if (!is_active(node_id) || reached[node_id] != 0) {
                    continue;
                }

                reached[node_id] = 1;
                frontier.emplace_back(node_id);
            }
        }

        return reached;
    }

    const DistributedGraph& graph;

    mpi_rank_type my_rank{};
    node_id_type number_local_nodes{};

    std::vector<global_node_id_type> prefix_distribution{};
    global_node_id_type total_number_nodes{};

    std::vector<std::uint8_t> assigned{};
    std::vector<global_node_id_type> labels{};

    std::vector<arc_id_type> remaining_in_degrees{};
    std::vector<arc_id_type> remaining_out_degrees{};
};
