/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NetworkMotifs.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/counting/NodeDistributionCounter.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaAccumulateVariant.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaFairVariant.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaFewerArcsVariant.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaFewerMessagesVariant.h"
#include "metrics/local_structure/NetworkMotifsVariants/RmaVariant.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <fmt/format.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>
#include <mpi-wrapper/patterns/comm_patterns/Questions.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

std::array<long double, 14> NetworkMotifs::compute_network_triple_motifs(const DistributedGraph& graph, const NetworkMotifAlgorithm algorithm) {
    auto timer_compute = utility::RegionTimer{ "network-motifs-compute" };

    const auto motif_fraction = [&graph, algorithm]() -> std::array<long double, 14> {
        switch (algorithm) {
        case NetworkMotifAlgorithm::Questions:
            return compute_motifs_with_questions(graph);
        case NetworkMotifAlgorithm::Rma:
            return RmaVariant::all_compute_motifs(graph);
        case NetworkMotifAlgorithm::RmaAccumulate:
            return RmaAccumulateVariant::all_compute_motifs(graph);
        case NetworkMotifAlgorithm::RmaFair:
            return RmaFairVariant::all_compute_motifs(graph);
        case NetworkMotifAlgorithm::RmaFewerArcs:
            return RmaFewerArcsVariant::all_compute_motifs(graph);
        case NetworkMotifAlgorithm::RmaFewerMessages:
            return RmaFewerMessagesVariant::all_compute_motifs(graph);
        default:
            utility::Exception::fail("NetworkMotifs::compute_network_triple_motifs: the algorithm is unknown");
        }
    }();

    timer_compute.stop();

    mpiPP::MPIRegionTimerReport::reduce_and_print("NetworkMotifs");
    utility::RegionTimer::init();

    return motif_fraction;
}

std::array<long double, 14> NetworkMotifs::compute_motifs_with_questions(const DistributedGraph& graph) {
    // Testing function parameters
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
    const auto number_local_nodes = graph.get_number_local_nodes();

    auto timer_compute_codes = utility::RegionTimer{ "network-motifs-compute-codes" };

    // Calculate the networkMotifs
    const auto collect_possible_network_motifs_one_node = [my_rank, &graph](const node_id_type node_local_index) {
        using ResultType = mpiPP::comm_patterns::GenerateQuestionsFunction<ThreeMotifStructure, node_id_type>::result_type;

        const auto out_arcs = graph.get_out_arcs(my_rank, node_local_index);
        const auto in_arcs = graph.get_in_arcs(my_rank, node_local_index);

        auto this_node_possible_motifs = ResultType();
        this_node_possible_motifs.reserve(out_arcs.size() * in_arcs.size());

        // Compute an array of unique connected nodes with an bool pair wether the connecting arcs are out,
        // in or both
        auto adjacent_nodes_list = std::vector<std::pair<std::pair<mpi_rank_type, node_id_type>, std::pair<bool, bool>>>{};
        adjacent_nodes_list.reserve(out_arcs.size() + in_arcs.size());

        auto adjacent_nodes_to_index = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, std::size_t, utility::hash<std::pair<mpi_rank_type, node_id_type>>>{};
        adjacent_nodes_to_index.reserve(out_arcs.size() + in_arcs.size());

        for (const auto& [target_rank, target_id, _] : out_arcs) {
            if (target_rank == my_rank && target_id == node_local_index) {
                continue;
            }

            const auto identifier = std::pair{ target_rank, target_id };

            if (const auto entry = adjacent_nodes_to_index.find(identifier); entry == adjacent_nodes_to_index.end()) {
                adjacent_nodes_list.emplace_back(identifier, std::pair{ true, false });
                adjacent_nodes_to_index[identifier] = adjacent_nodes_list.size() - 1;
            }
        }

        for (const auto& [source_rank, source_id, _] : in_arcs) {
            if (source_rank == my_rank && source_id == node_local_index) {
                continue;
            }

            const auto identifier = std::pair{ source_rank, source_id };
            if (const auto entry = adjacent_nodes_to_index.find(identifier); entry == adjacent_nodes_to_index.end()) {
                adjacent_nodes_list.emplace_back(identifier, std::pair{ false, true });
            } else {
                const auto pair_index = entry->second;
                assert(pair_index < adjacent_nodes_list.size());

                auto& [first, second] = adjacent_nodes_list[pair_index].second;
                assert(first);
                second = true;
            }
        }

        // Iterate over all node triples connected to node to compute possible motif types
        for (auto i = std::size_t{ 0 }; i < adjacent_nodes_list.size(); ++i) {
            const auto& [node_outer, arcs_out_in_to_node_outer] = adjacent_nodes_list[i];
            const auto& [node_2_exists_out_edge, node_2_exists_in_edge] = arcs_out_in_to_node_outer;

            for (auto j = std::size_t{ 0 }; j < adjacent_nodes_list.size(); ++j) {
                if (i == j) {
                    continue;
                }

                const auto& [node_inner, arcs_out_in_to_node_inner] = adjacent_nodes_list[j];
                const auto& [node_inner_rank, node_inner_id] = node_inner;
                const auto& [node_3_exists_out_edge, node_3_exists_in_edge] = arcs_out_in_to_node_inner;

                auto motif_structure = ThreeMotifStructure{ .node_3_rank = node_inner_rank, .node_3_local = node_inner_id };

                auto exists_arc_bit_array = 0U;
                exists_arc_bit_array |= node_2_exists_out_edge ? 1U : 0U;
                exists_arc_bit_array |= node_2_exists_in_edge ? 2U : 0U;
                exists_arc_bit_array |= node_3_exists_out_edge ? 4U : 0U;
                exists_arc_bit_array |= node_3_exists_in_edge ? 8U : 0U;

                switch (exists_arc_bit_array) {
                case 5:
                    // three node motif 3 & 5 & 8 (0101)
                    motif_structure.set_motif_types({ 3, 5, 8 });
                    break;
                case 6:
                    // three node motif 10 (0110)
                    motif_structure.set_motif_types({ 10 });
                    break;
                case 7:
                    // three node motif 6 (0111)
                    motif_structure.set_motif_types({ 6 });
                    break;
                case 9:
                    // three node motif 2 & 7 (1001)
                    motif_structure.set_motif_types({ 2, 7 });
                    break;
                case 10:
                    // three node motif 1 & 11 (1010)
                    motif_structure.set_motif_types({ 1, 11 });
                    break;
                case 11:
                    // three node motif 4 (1011)
                    motif_structure.set_motif_types({ 4 });
                    break;
                case 13:
                    continue;
                case 14:
                    continue;
                case 15:
                    // three node motif 9 & 12 & 13 (1111)
                    motif_structure.set_motif_types({ 9, 12, 13 });
                    break;
                default:
                    assert(false);
                }

                this_node_possible_motifs.emplace_back(node_outer.first, node_outer.second, motif_structure);
            }
        }

        return this_node_possible_motifs;
    };

    const auto evaluate_correct_network_motifs_one_node = [my_rank, &graph](const node_id_type node_local_index,
                                                                            ThreeMotifStructure possible_motif) {
        const auto out_arcs = graph.get_out_arcs(my_rank, node_local_index);
        const auto in_arcs = graph.get_in_arcs(my_rank, node_local_index);

        // Compute an map of unique connected nodes with an bool pair wether the connecting arcs are
        // out, in or both
        auto adjacent_nodes = std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>, utility::hash<std::pair<std::uint64_t, std::uint64_t>>>{};

        for (const auto& [target_rank, target_id, _] : out_arcs) {
            const auto node_key = std::pair<std::uint64_t, std::uint64_t>(target_rank, target_id);
            auto& value = adjacent_nodes[node_key];
            value.first = true;
        }

        for (const auto& [source_rank, source_id, _] : in_arcs) {
            const auto node_key = std::pair<std::uint64_t, std::uint64_t>(source_rank, source_id);
            auto& value = adjacent_nodes[node_key];
            value.second = true;
        }

        // Being on node 2 of the possible_motif check which connection to node 3 exist and decide on
        // the correct motif
        const auto node_3_key = std::pair<std::uint64_t, std::uint64_t>(possible_motif.node_3_rank, possible_motif.node_3_local);
        auto arcs_connected = std::pair<bool, bool>{ false, false };

        if (adjacent_nodes.find(node_3_key) != adjacent_nodes.end()) {
            arcs_connected = adjacent_nodes[node_3_key];
        }

        const auto exists_arc_node2_to_node3 = arcs_connected.first;
        const auto exists_arc_node3_to_node2 = arcs_connected.second;

        if (exists_arc_node2_to_node3 && exists_arc_node3_to_node2) {
            // arcs between node 2 and 3 in both directions
            // maintain motifs 8,10,11,13
            possible_motif.unset_motif_types({ 1, 2, 3, 4, 5, 6, 7, 9, 12 });

        } else if (exists_arc_node2_to_node3 && !exists_arc_node3_to_node2) {
            // only arc from node 2 to node 3
            // maintain motifs 5,7
            possible_motif.unset_motif_types({ 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13 });

        } else if (!exists_arc_node2_to_node3 && exists_arc_node3_to_node2) {
            // only arc from node 3 to node 2
            // maintain motifs 12
            possible_motif.unset_motif_types({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13 });

        } else {
            // no arcs between node 2 and 3
            // maintain motifs 1,2,3,4,6,9
            possible_motif.unset_motif_types({ 5, 7, 8, 10, 11, 12, 13 });
        }

        assert(possible_motif.check_validity());
        return possible_motif;
    };

    const auto three_motif_results = mpiPP::comm_patterns::node_to_node_question<ThreeMotifStructure, ThreeMotifStructure, node_id_type>(
        number_local_nodes, collect_possible_network_motifs_one_node, evaluate_correct_network_motifs_one_node);

    timer_compute_codes.stop();

    auto timer_collect_number = utility::RegionTimer{ "network-motifs-collect-number" };

    // Count the number of motifs locally
    auto motif_type_count = std::array<std::uint64_t, 14>{};

    for (auto node_local_index = node_id_type{ 0 }; node_local_index < number_local_nodes; ++node_local_index) {
        const auto& this_node_motifs_results = three_motif_results.get_answers_of_questioner_node(node_local_index);

        for (const auto& one_motif : this_node_motifs_results) {
            for (auto motif_type = 1U; motif_type < 14U; motif_type++) {
                if (one_motif.is_motif_type_set(motif_type)) {
                    motif_type_count[motif_type]++;
                }
            }
        }
    }

    timer_collect_number.stop();

    auto timer_collect_number_global = utility::RegionTimer{ "network-motifs-collect-number-global" };

    // Collect the number of motifs globally
    auto motif_type_count_total = mpiPP::MPIReductions::reduce_componentwise_sum(motif_type_count);

    // Reduce the motifs that were counted multiple times due to their invariant nature
    if (my_rank == 0) {
        // Order invariant motifs were counted two times each
        if (motif_type_count_total[1] % 2 != 0) {
            spdlog::warn("Result for motif 1 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[1] /= 2;

        if (motif_type_count_total[3] % 2 != 0) {
            spdlog::warn("Result for motif 3 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[3] /= 2;

        if (motif_type_count_total[8] % 2 != 0) {
            spdlog::warn("Result for motif 8 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[8] /= 2;

        if (motif_type_count_total[9] % 2 != 0) {
            spdlog::warn("Result for motif 9 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[9] /= 2;

        if (motif_type_count_total[11] % 2 != 0) {
            spdlog::warn("Result for motif 11 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[11] /= 2;

        // Rotational invariant motifs were counted three times each
        if (motif_type_count_total[7] % 3 != 0) {
            spdlog::warn("Result for motif 7 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[7] /= 3;

        // Order and Rotational invariant motifs were counted six times each
        if (motif_type_count_total[13] % 6 != 0) {
            spdlog::warn("Result for motif 13 may be unprecise due to inconsistent arcs (arc exists only as in- or out-arc but not as both)");
        }
        motif_type_count_total[13] /= 6;
    }

    timer_collect_number_global.stop();

    auto timer_compute_distribution = utility::RegionTimer{ "network-motifs-compute-distribution" };

    // Compute the resulting array of motif numbers
    auto motif_fraction = std::array<long double, 14>{};
    if (my_rank == 0) {
        const auto total_number_of_motifs = std::accumulate(motif_type_count_total.begin(), motif_type_count_total.end(), std::uint64_t{ 0 });
        motif_fraction[0] = static_cast<long double>(total_number_of_motifs);

        for (auto motif_type = 1U; motif_type < 14U; motif_type++) {
            motif_fraction[motif_type] = static_cast<long double>(motif_type_count_total[motif_type]) / static_cast<long double>(total_number_of_motifs);
        }
    }

    timer_compute_distribution.stop();

    return motif_fraction;
}

std::array<long double, 14> NetworkMotifs::compute_network_triple_motifs_sequential(const DistributedGraph& graph) {
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto number_nodes_of_ranks = NodeDistributionCounter::all_count_node_distribution(graph);

    if (!mpiPP::MPIInfo::is_root_rank()) {
        mpiPP::MPISynchronization::barrier();
        return {};
    }

    // Prepare structure
    auto motif_type_count = std::vector<std::uint64_t>(14, 0);

    for (auto current_rank = mpi_rank_type{ 0 }; current_rank < number_ranks; ++current_rank) {
        for (auto current_node = node_id_type{ 0 }; current_node < number_nodes_of_ranks[utility::safe_cast<std::size_t>(current_rank)]; ++current_node) {

            // Gather information of adjacent nodes
            const auto out_arcs = graph.get_out_arcs(current_rank, current_node);
            const auto in_arcs = graph.get_in_arcs(current_rank, current_node);

            auto adjacent_nodes = std::unordered_map<std::pair<mpi_rank_type, node_id_type>, std::pair<bool, bool>, utility::hash<std::pair<mpi_rank_type, node_id_type>>>{};

            for (const auto& [target_rank, target_id, _] : out_arcs) {
                auto& [first, second] = adjacent_nodes[{ target_rank, target_id }];
                assert(!first);
                first = true;
            }

            for (const auto& [source_rank, source_id, _] : in_arcs) {
                auto& [first, second] = adjacent_nodes[{ source_rank, source_id }];
                assert(!second);
                second = true;
            }

            for (auto iter_outer = adjacent_nodes.begin(); iter_outer != adjacent_nodes.end(); ++iter_outer) {
                // Prrevent overlapping of outer and inner
                // -> consider only each tripple-node-set (ignore permutations)
                const auto node_outer_key = iter_outer->first;
                auto iter_inner = iter_outer;
                iter_inner++;

                for (; iter_inner != adjacent_nodes.end(); ++iter_inner) {
                    const auto node_inner_key = iter_inner->first;

                    if (node_inner_key != node_outer_key) {
                        // Exclude nodes with self-referencing arcs
                        if ((current_rank == node_outer_key.first && current_node == node_outer_key.second) || (current_rank == node_inner_key.first && current_node == node_inner_key.second) || (node_outer_key.first == node_inner_key.first && node_outer_key.second == node_inner_key.second)) {
                            continue;
                        }

                        ThreeMotifStructure motif_structure;
                        motif_structure.node_3_rank = node_inner_key.first;
                        motif_structure.node_3_local = node_inner_key.second;

                        const bool exists_arc_node1_to_node2 = iter_outer->second.first;
                        const bool exists_arc_node2_to_node1 = iter_outer->second.second;
                        const bool exists_arc_node1_to_node3 = iter_inner->second.first;
                        const bool exists_arc_node3_to_node1 = iter_inner->second.second;

                        auto exists_arc_bit_array = 0U;
                        exists_arc_bit_array |= exists_arc_node1_to_node2 ? 1U : 0U;
                        exists_arc_bit_array |= exists_arc_node2_to_node1 ? 2U : 0U;
                        exists_arc_bit_array |= exists_arc_node1_to_node3 ? 4U : 0U;
                        exists_arc_bit_array |= exists_arc_node3_to_node1 ? 8U : 0U;

                        const std::uint16_t exists_arc_bit_array_updated = update_arc_bit_array(
                            graph, utility::safe_cast<std::uint16_t>(exists_arc_bit_array), node_outer_key.first, node_outer_key.second, motif_structure.node_3_rank, motif_structure.node_3_local);

                        switch (exists_arc_bit_array) {
                        case 10:
                            // three node motif 1 & 11 (0101)
                            if (exists_arc_bit_array_updated == 10) {
                                motif_structure.set_motif_types({ 1 });
                            } else if (exists_arc_bit_array_updated == 58) {
                                motif_structure.set_motif_types({ 11 });
                            }
                            break;

                        case 5:
                            // three node motif 3 & 8 (1010)
                            if (exists_arc_bit_array_updated == 5) {
                                motif_structure.set_motif_types({ 3 });
                            } else if (exists_arc_bit_array_updated == 53) {
                                motif_structure.set_motif_types({ 8 });
                            }
                            break;

                        case 6:
                        case 9:
                            // three node motif 2 & 5 & 7 & 10 (0110, 1001)
                            if (exists_arc_bit_array_updated == 6 || exists_arc_bit_array_updated == 9) {
                                motif_structure.set_motif_types({ 2 });
                            } else if (exists_arc_bit_array_updated == 22 || exists_arc_bit_array_updated == 41) {
                                motif_structure.set_motif_types({ 5 });
                            } else if (exists_arc_bit_array_updated == 38 || exists_arc_bit_array_updated == 25) {
                                motif_structure.set_motif_types({ 7 });
                            } else if (exists_arc_bit_array_updated == 54 || exists_arc_bit_array_updated == 57) {
                                motif_structure.set_motif_types({ 10 });
                            } else {
                                spdlog::error("case 6/9 -> bitArray = {}", exists_arc_bit_array_updated);
                            }
                            break;

                        case 14:
                        case 11:
                            // three node motif 4 (0111, 1101)
                            if (exists_arc_bit_array_updated == 14 || exists_arc_bit_array_updated == 11) {
                                motif_structure.set_motif_types({ 4 });
                            }
                            break;

                        case 7:
                        case 13:
                            // three node motif 6 (1110, 1011)
                            if (exists_arc_bit_array_updated == 7 || exists_arc_bit_array_updated == 13) {
                                motif_structure.set_motif_types({ 6 });
                            }
                            break;

                        case 15:
                            // three node motif 9 & 12 & 13 (1111)
                            if (exists_arc_bit_array_updated == 15) {
                                motif_structure.set_motif_types({ 9 });
                            } else if (exists_arc_bit_array_updated == 31 || exists_arc_bit_array_updated == 47) {
                                motif_structure.set_motif_types({ 12 });
                            } else if (exists_arc_bit_array_updated == 63) {
                                motif_structure.set_motif_types({ 13 });
                            } else {
                                spdlog::error("case 15 -> bitArray = {}", exists_arc_bit_array_updated);
                            }
                            break;

                        default:
                            break;
                        }

                        // Count every motif
                        for (auto motif_type = 1U; motif_type < 14U; ++motif_type) {
                            if (motif_structure.is_motif_type_set(motif_type)) {
                                ++motif_type_count[motif_type];
                            }
                        }
                    }
                }
            }
        }
    }

    // Rotational invariant motifs where counted three times each
    if (motif_type_count[7] % 3 != 0) {
        spdlog::error("motif_type_count[7]%3 != 0 ==> {}", motif_type_count[7]);
    }

    motif_type_count[7] /= 3;

    if (motif_type_count[13] % 3 != 0) {
        spdlog::error("motif_type_count[13]%3 != 0 ==> {}", motif_type_count[13]);
    }

    motif_type_count[13] /= 3;

    auto motif_fraction = std::array<long double, 14>{};
    const auto total_number_of_motifs = std::accumulate(motif_type_count.begin(), motif_type_count.end(), std::uint64_t{ 0 });
    motif_fraction[0] = static_cast<long double>(total_number_of_motifs);

    for (auto motif_type = 1U; motif_type < 14U; motif_type++) {
        motif_fraction[motif_type] = static_cast<long double>(motif_type_count[motif_type]) / static_cast<long double>(total_number_of_motifs);
    }

    mpiPP::MPISynchronization::barrier();
    return motif_fraction;
}

std::uint16_t NetworkMotifs::update_arc_bit_array(const DistributedGraph& graph, std::uint16_t exists_arc_bit_array, const mpi_rank_type node_2_rank,
                                                  const node_id_type node_2_local, const mpi_rank_type node_3_rank, const node_id_type node_3_local) {
    const auto out_arcs = graph.get_out_arcs(node_2_rank, node_2_local);
    const auto in_arcs = graph.get_in_arcs(node_2_rank, node_2_local);

    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>, utility::hash<std::pair<std::uint64_t, std::uint64_t>>> adjacent_nodes;

    for (const auto& [target_rank, target_id, _] : out_arcs) {
        const std::pair<std::uint64_t, std::uint64_t> node_key(target_rank, target_id);
        std::pair<bool, bool>& value = adjacent_nodes[node_key];
        assert(!value.first);
        value.first = true;
    }

    for (const auto& [source_rank, source_id, _] : in_arcs) {
        const std::pair<std::uint64_t, std::uint64_t> node_key(source_rank, source_id);
        std::pair<bool, bool>& value = adjacent_nodes[node_key];
        assert(!value.second);
        value.second = true;
    }

    const std::pair<std::uint64_t, std::uint64_t> node_3_key(node_3_rank, node_3_local);
    const std::pair<bool, bool>& value = adjacent_nodes[node_3_key];

    const bool exists_arc_node2_to_node3 = value.first;
    const bool exists_arc_node3_to_node2 = value.second;

    exists_arc_bit_array |= exists_arc_node2_to_node3 ? std::uint16_t{ 16 } : std::uint16_t{ 0 };
    exists_arc_bit_array |= exists_arc_node3_to_node2 ? std::uint16_t{ 32 } : std::uint16_t{ 0 };

    return exists_arc_bit_array;
}

void NetworkMotifs::ThreeMotifStructure::self_test() {
    for (auto motif = 1U; motif < 14; ++motif) {
        set_motif_types({ motif });
        unset_motif_types({ motif });
        assert(motif_type_bit_array == 0);
    }

    for (auto motif = 1U; motif < 14; ++motif) {
        set_motif_types({ motif });
    }

    for (auto motif = 1U; motif < 14; ++motif) {
        unset_motif_types({ motif });
    }

    assert(motif_type_bit_array == 0);
}

void NetworkMotifs::ThreeMotifStructure::set_motif_types(const std::vector<unsigned int>& motif_types) {
    for (const auto motif_type : motif_types) {
        assert(motif_type >= 1 && motif_type < 14);
        motif_type_bit_array |= utility::safe_cast<std::uint16_t>(1U << motif_type);
    }
}

void NetworkMotifs::ThreeMotifStructure::unset_motif_types(const std::vector<unsigned int>& motif_types) {
    for (const auto motif_type : motif_types) {
        assert(motif_type >= 1 && motif_type < 14);
        motif_type_bit_array &= static_cast<std::uint16_t>(~(1U << motif_type));
    }
}

bool NetworkMotifs::ThreeMotifStructure::is_motif_type_set(const unsigned int motif_type) const {
    assert(motif_type >= 1 && motif_type < 14);
    return (motif_type_bit_array & (1U << motif_type)) != 0;
}

void NetworkMotifs::ThreeMotifStructure::print_out() const {
    fmt::print("---");
    for (auto i = 1U; i < 14U; i++) {
        fmt::print("{}", is_motif_type_set(i));
    }
    fmt::print("--------------------\n");
}

bool NetworkMotifs::ThreeMotifStructure::check_validity() const {
    auto res = (motif_type_bit_array != 0U) && ((motif_type_bit_array & (motif_type_bit_array - 1U)) == 0);
    res |= motif_type_bit_array == 0;
    if (!res) {
        print_out();
    }
    return res;
}
