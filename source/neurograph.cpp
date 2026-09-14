/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "graph/DistributedGraph.h"
#include "metrics/community/AreaConnectivity.h"
#include "metrics/community/Modularity.h"
#include "metrics/connectivity/StronglyConnectedComponents.h"
#include "metrics/counting/InArcCounter.h"
#include "metrics/counting/InWeightCounter.h"
#include "metrics/counting/NodeCounter.h"
#include "metrics/counting/OutArcCounter.h"
#include "metrics/counting/OutWeightCounter.h"
#include "metrics/degree/InDegree.h"
#include "metrics/degree/InDegreeHistogram.h"
#include "metrics/degree/OutDegree.h"
#include "metrics/degree/OutDegreeHistogram.h"
#include "metrics/flow/MaximumFlow.h"
#include "metrics/geometry/AllPairsDistances.h"
#include "metrics/geometry/AllPairsDistancesHistogram.h"
#include "metrics/geometry/ArcLength.h"
#include "metrics/geometry/ArcLengthHistogram.h"
#include "metrics/local_structure/Assortativity.h"
#include "metrics/local_structure/Clustering.h"
#include "metrics/local_structure/NetworkMotifs.h"
#include "metrics/local_structure/Reciprocity.h"
#include "metrics/local_structure/RichClub.h"
#include "metrics/local_structure/Transitivity.h"
#include "metrics/paths/AllPairsShortestPath.h"
#include "metrics/paths/BetweennessCentrality.h"
#include "metrics/paths/DiameterApproximation.h"
#include "utility/MetricStore.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <mpi-wrapper/MPIWrapper.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/instrumentation/MPIProgress.h>
#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>

#include <omp.h>

#include <spdlog/spdlog.h>

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// Struct to store all the CLI arguments bundled up
//	so we dont need to give them sperately as arguments
struct CLIArguments {
    std::filesystem::path input_directory;

    bool enable_node_count;
    bool enable_arc_count;
    bool enable_weight_count;

    bool enable_degree_extremes;
    bool enable_degree_histogram_width;
    arc_id_type bin_width_degree_histogram = 1;
    bool enable_degree_histogram_count;
    // std::size_t bin_count_degree_histogram = 1; This is always set such that the bins have width 1

    bool enable_pair_distances_average;
    bool enable_pair_distances_histogram_width;
    double bin_width_pair_distances_histogram = 1.0;
    bool enable_pair_distances_histogram_count;
    std::size_t bin_count_pair_distances_histogram = 1;

    bool enable_arc_length_average;
    bool enable_arc_length_histogram_width;
    double bin_width_arc_length_histogram = 1.0;
    bool enable_arc_length_histogram_count;
    std::size_t bin_count_arc_length_histogram = 1;

    bool enable_all_pairs_shortest_path;
    SsspAlgorithm apsp_algorithm = SsspAlgorithm::Dijkstra;
    distance_type apsp_delta = 1;
    DeltaSteppingEpochType apsp_delta_epoch = DeltaSteppingEpochType::PushCheck;
    DijkstraQueueType dijkstra_queue = DijkstraQueueType::PriorityQueue;
    bool enable_all_pairs_shortest_path_inverse;
    bool enable_clustering_coefficient;
    bool enable_betweenness_centrality_average;
    BetweennessCentralityAlgorithm centrality_algorithm = BetweennessCentralityAlgorithm::Brandes;
    bool enable_area_connectivity;
    bool enable_modularity;
    ModularityAlgorithm modularity_algorithm = ModularityAlgorithm::Louvain;
    bool enable_network_motifs;
    NetworkMotifAlgorithm network_motif_algorithm = NetworkMotifAlgorithm::Questions;
    bool enable_assortativity;
    bool enable_reciprocity;
    bool enable_rich_club;
    bool enable_transitivity;
    bool enable_diameter_approximation;

    bool enable_strongly_connected_components;
    SccAlgorithm scc_algorithm = SccAlgorithm::Multistep;
    global_node_id_type scc_tail_threshold = 0;

    bool enable_maximum_flow;
    std::vector<NodeIdentifier> max_flow_sources;
    std::vector<NodeIdentifier> max_flow_sinks;
    MaxFlowAlgorithm max_flow_algorithm = MaxFlowAlgorithm::Dinic;

    bool remove_autapses;
    bool undirected;
    bool one_weight;

    std::string prefix;
    std::filesystem::path output_file;
};

static void calculate_metrics(const CLIArguments& args) {
    auto dg = DistributedGraph::construct_graph(args.input_directory, args.remove_autapses, args.undirected, args.one_weight, args.prefix);

    auto ms = MetricStore{};

    mpiPP::MPISynchronization::barrier();

    if (mpiPP::MPIInfo::is_root_rank()) {
        fmt::print("All {} ranks finished loading their local data!\n", mpiPP::MPIInfo::get_number_ranks_cast());
    }

    mpiPP::MPISynchronization::barrier();

    if (args.enable_node_count) {
        const auto number_total_nodes = NodeCounter::count_nodes(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_global_number_nodes(number_total_nodes);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_arc_count) {
        const auto number_total_in_arcs = InArcCounter::count_in_arcs(dg);
        const auto number_total_out_arcs = OutArcCounter::count_out_arcs(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_number_in_arcs(number_total_in_arcs);
            ms.set_number_out_arcs(number_total_out_arcs);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_weight_count) {
        const auto weight_total_in_arcs = InWeightCounter::weigh_in_arcs(dg);
        const auto weight_total_out_arcs = OutWeightCounter::weigh_out_arcs(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_weight_in_arcs(weight_total_in_arcs);
            ms.set_weight_out_arcs(weight_total_out_arcs);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_degree_extremes) {
        const auto in_extremes = InDegree::compute_extreme_degrees(dg);
        const auto out_extremes = OutDegree::compute_extreme_degrees(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_extreme_in_degrees(in_extremes);
            ms.set_extreme_out_degrees(out_extremes);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_degree_histogram_width) {
        const auto width = args.bin_width_degree_histogram;
        auto in_degree_hist = InDegreeHistogram::compute_fixed_bin_width(dg, width);
        auto out_degree_hist = OutDegreeHistogram::compute_fixed_bin_width(dg, width);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_in_degree_histogram_width(std::move(in_degree_hist));
            ms.set_out_degree_histogram_width(std::move(out_degree_hist));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_degree_histogram_count) {
        const auto max_in_degree = InDegree::compute_maximum_degree(dg);
        const auto max_out_degree = OutDegree::compute_maximum_degree(dg);
        auto in_degree_hist = InDegreeHistogram::compute_fixed_bin_count(dg, 0, max_in_degree + 1, max_in_degree + 1);
        auto out_degree_hist = OutDegreeHistogram::compute_fixed_bin_count(dg, 0, max_out_degree + 1, max_out_degree + 1);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_in_degree_histogram_count(std::move(in_degree_hist));
            ms.set_out_degree_histogram_count(std::move(out_degree_hist));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_pair_distances_average) {
        const auto average_pair_distance = AllPairsDistances::compute_average_pair_distance(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_average_pair_distances(average_pair_distance);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_pair_distances_histogram_width) {
        const auto width = args.bin_width_pair_distances_histogram;
        auto distances = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_width(dg, width);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_pair_distances_histogram_width(std::move(distances));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_pair_distances_histogram_count) {
        const auto max_distance = AllPairsDistances::compute_maximum_pair_distance(dg);
        const auto count = args.bin_count_pair_distances_histogram;
        auto distances = AllPairsDistancesHistogram::compute_pair_distances_fixed_bin_count(dg, 0, max_distance + 1e-5, count);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_pair_distances_histogram_count(std::move(distances));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_arc_length_average) {
        const auto average_arc_length = ArcLength::compute_average_arc_length(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_average_arc_length(average_arc_length);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_arc_length_histogram_width) {
        const auto width = args.bin_width_pair_distances_histogram;
        auto histogram_wid = ArcLengthHistogram::compute_histogram_fixed_bin_width(dg, width);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_arc_length_histogram_width(std::move(histogram_wid));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_arc_length_histogram_count) {
        const auto max_length = ArcLength::all_compute_maximum_arc_length(dg);
        const auto count = args.bin_count_arc_length_histogram;
        auto histogram_cnt = ArcLengthHistogram::compute_histogram_fixed_bin_count(dg, 0, max_length + 1e-5, count);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_arc_length_histogram_count(std::move(histogram_cnt));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_all_pairs_shortest_path) {
        dg.lock_all_rma_windows();
        const auto& apsp_result = (args.apsp_algorithm == SsspAlgorithm::DeltaStepping)
            ? AllPairsShortestPath::compute_apsp_delta_stepping(dg, args.apsp_delta, args.apsp_delta_epoch)
            : AllPairsShortestPath::compute_apsp(dg, args.dijkstra_queue);
        dg.unlock_all_rma_windows();

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_all_pairs_shortest_paths_result(apsp_result);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_all_pairs_shortest_path_inverse) {
        dg.lock_all_rma_windows();
        const auto& apsp_result = AllPairsShortestPath::compute_apsp_inverse(dg, args.dijkstra_queue);
        dg.unlock_all_rma_windows();

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_all_pairs_shortest_paths_inverse_result(apsp_result);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_clustering_coefficient) {
        const auto average_cluster_coefficient = Clustering::compute_average_clustering_coefficient(dg);
        const auto average_cluster_coefficient_2 = Clustering::compute_average_clustering_coefficient_2(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_average_clustering_coefficient(average_cluster_coefficient);
            ms.set_average_clustering_coefficient_2(average_cluster_coefficient_2);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_betweenness_centrality_average) {
        const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg, args.centrality_algorithm);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_average_betweenness_centrality(average_betweenness_centrality);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_area_connectivity) {
        auto area_connectivity = AreaConnectivity::compute_area_connectivity_strength(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_area_connectivity_map(std::move(area_connectivity));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_modularity) {
        const auto modularity = Modularity::compute_modularity(dg, args.modularity_algorithm);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_modularity(modularity);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_network_motifs) {
        const auto& motifs = NetworkMotifs::compute_network_triple_motifs(dg, args.network_motif_algorithm);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_motifs(motifs);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_assortativity) {
        const auto& assortativity = Assortativity::compute_assortativity(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_assortativity(assortativity);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_reciprocity) {
        const auto& reciprocity = Reciprocity::compute_reciprocity(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_reciprocity(reciprocity);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_rich_club) {
        auto rich_club = RichClub::compute_rich_club_coefficients(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_rich_club(std::move(rich_club));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_transitivity) {
        const auto& transitivity = Transitivity::compute_transitivity(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_transitivity(transitivity);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_diameter_approximation) {
        const auto approximate_diameter = DiameterApproximation::compute_approximation(dg, 1);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_approximate_diameter(approximate_diameter);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_strongly_connected_components) {
        auto components = StronglyConnectedComponents::compute_strongly_connected_components(dg, args.scc_algorithm, args.scc_tail_threshold);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_strongly_connected_components(std::move(components));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_maximum_flow) {
        const auto max_flow = MaximumFlow::compute_maximum_flow(dg, args.max_flow_sources, args.max_flow_sinks, args.max_flow_algorithm);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_maximum_flow(max_flow);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (mpiPP::MPIInfo::is_root_rank()) {
        ms.output(args.output_file);
    }
}

// Parses a single "rank:node_id" entry (as used by --max-flow-sources/--max-flow-sinks)
static std::optional<NodeIdentifier> parse_node_identifier(const std::string& entry) {
    const auto separator = entry.find(':');
    if (separator == std::string::npos) {
        return std::nullopt;
    }

    auto rank = mpi_rank_type{};
    auto node_id = node_id_type{};

    const auto* const rank_end = entry.data() + separator;
    const auto* const node_id_end = entry.data() + entry.size();

    const auto rank_result = std::from_chars(entry.data(), rank_end, rank);
    const auto node_id_result = std::from_chars(entry.data() + separator + 1, node_id_end, node_id);

    if (rank_result.ec != std::errc{} || rank_result.ptr != rank_end || node_id_result.ec != std::errc{} || node_id_result.ptr != node_id_end) {
        return std::nullopt;
    }

    return NodeIdentifier{ rank, node_id };
}

static void print_arguments(const int argument_count, char* arguments[]) {
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    fmt::print("Arguments are:\n{}\n\n", fmt::join(std::span<char*>{ arguments, utility::safe_cast<std::size_t>(argument_count) }, " "));
}

int main(const int argument_count, char* arguments[]) {
    auto app = CLI::App{ "" };
    auto args = CLIArguments{};

    auto* opt_input_directory = app.add_option("--input", args.input_directory, "The directory that contains the input files.")->required();
    opt_input_directory->check(CLI::ExistingDirectory);

    auto selected_algorithms = std::vector<std::string>{};

    app.add_option("--func", selected_algorithms,
                 "Select what functions to run as a comma-seperated list:\n"
                 "node-count,\n"
                 "arc-count,\n"
                 "weight-count,\n"
                 "degree-extremes,\n"
                 "degree-histogram-width,\n"
                 "degree-histogram-count,\n"
                 "arc-length-average,\n"
                 "arc-length-histogram-width,\n"
                 "arc-length-histogram-count,\n"
                 "pair-distances-average,\n"
                 "pair-distances-histogram-width,\n"
                 "pair-distances-histogram-count,\n"
                 "avg-apsp,\n"
                 "avg-apsp-inv,\n"
                 "avg-cluster,\n"
                 "avg-centrality,\n"
                 "area-connec,\n"
                 "modularity,\n"
                 "tri-motifs,\n"
                 "assortativity,\n"
                 "reciprocity,\n"
                 "rich-club,\n"
                 "transitivity,\n"
                 "dia-approx,\n"
                 "scc,\n"
                 "max-flow")
        ->delimiter(',')
        ->allow_extra_args(true);

    int number_threads = 1;
    app.add_option("--threads", number_threads, "Number of OMP threads to use for some calculations (default: 1)");

    auto* flag_undirected = app.add_flag("--undirected", "Treats all synapses as undirected (sums weight)");
    auto* flag_one_weight = app.add_flag("--all-weights-one", "Treats all weights as 1 (overwrites sum regarding undirected)");
    auto* flag_remove_autapses = app.add_flag("--remove-autapses", "Removes autapses (synapses from a neuron to itself)");

    app.add_option("--prefix", args.prefix);

    app.add_option("--bin-width-degree-histogram", args.bin_width_degree_histogram);
    // app.add_option("--bin-count-degree-histogram", args.bin_count_degree_histogram);

    app.add_option("--bin-width-pair-distances-histogram", args.bin_width_pair_distances_histogram);
    app.add_option("--bin-count-pair-distances-histogram", args.bin_count_pair_distances_histogram);

    app.add_option("--bin-width-arc-length-histogram", args.bin_width_arc_length_histogram);
    app.add_option("--bin-count-arc-length-histogram", args.bin_count_arc_length_histogram);

    auto apsp_algorithm_arg = std::string{ "dijkstra" };
    app.add_option("--apsp-algorithm", apsp_algorithm_arg, "Algorithm used for avg-apsp: dijkstra (default) or delta-stepping");

    auto apsp_delta_epoch_arg = std::string{ "push-check" };
    app.add_option("--apsp-delta-epoch", apsp_delta_epoch_arg,
                   "Delta-stepping epoch strategy used when --apsp-algorithm is delta-stepping:\n"
                   "push-brute-force, push-check (default), push-short-long, push-inner-short, pull-model");

    app.add_option("--apsp-delta", args.apsp_delta, "Delta (bucket width) used when --apsp-algorithm is delta-stepping (default: 1)");

    auto dijkstra_queue_arg = std::string{ "priority-queue" };
    app.add_option("--dijkstra-queue", dijkstra_queue_arg,
                   "Vertex queue used by the Dijkstra-based avg-apsp and avg-apsp-inv:\n"
                   "priority-queue (default), fibonacci-heap, bucket-queue (not allowed for avg-apsp-inv)");

    auto centrality_algorithm_arg = std::string{ "brandes" };
    app.add_option("--centrality-algorithm", centrality_algorithm_arg, "Algorithm used for avg-centrality: brandes (default) or path-enumeration");

    auto modularity_algorithm_arg = std::string{ "louvain" };
    app.add_option("--modularity-algorithm", modularity_algorithm_arg,
                   "Community-detection algorithm used for modularity:\n"
                   "louvain (default), leiden, label-propagation, infomap.\n"
                   "Only louvain and leiden maximize the modularity itself, so the other two\n"
                   "usually report a lower value");

    auto network_motif_algorithm_arg = std::string{ "questions" };
    app.add_option("--tri-motifs-algorithm", network_motif_algorithm_arg,
                   "Algorithm used for tri-motifs:\n"
                   "questions (default), rma, rma-accumulate, rma-fair, rma-fewer-arcs, rma-fewer-messages.\n"
                   "All of them report the same fractions; questions asks the owner of a node whether two\n"
                   "of its neighbors are connected, the rma variants read those arcs themselves instead");

    auto scc_algorithm_arg = std::string{ "multistep" };
    app.add_option("--scc-algorithm", scc_algorithm_arg,
                   "Algorithm used for scc:\n"
                   "tarjan-gathered, forward-backward, coloring, multistep (default)");

    app.add_option("--scc-tail-threshold", args.scc_tail_threshold,
                   "How many nodes the multistep scc algorithm may gather on the root rank to finish\n"
                   "them sequentially (default: 0, which keeps every phase distributed)");

    auto max_flow_algorithm_arg = std::string{ "dinic" };
    app.add_option("--max-flow-algorithm", max_flow_algorithm_arg,
                   "Algorithm used for max-flow:\n"
                   "ford-fulkerson, edmonds-karp, dinic (default), orlin-king-rao-tarjan, distributed-push-relabel");

    auto max_flow_sources_arg = std::vector<std::string>{};
    app.add_option("--max-flow-sources", max_flow_sources_arg,
                   "Source nodes for max-flow, as a comma-separated list of rank:node_id pairs (required for max-flow)")
        ->delimiter(',');

    auto max_flow_sinks_arg = std::vector<std::string>{};
    app.add_option("--max-flow-sinks", max_flow_sinks_arg,
                   "Sink nodes for max-flow, as a comma-separated list of rank:node_id pairs (required for max-flow)")
        ->delimiter(',');

    auto output_directory = std::string{ "" };
    app.add_option("--output-directory", output_directory);

    auto* flag_disable_status = app.add_flag("--disable-status-report", "Disables reporting of the status");
    auto* flag_disable_timer = app.add_flag("--disable-timer-report", "Disables reporting of the timer");

    CLI11_PARSE(app, argument_count, arguments);

    omp_set_num_threads(std::max(1, number_threads));

    for (const auto& arg : selected_algorithms) {
        if (arg == "node-count") {
            args.enable_node_count = true;
        } else if (arg == "arc-count") {
            args.enable_arc_count = true;
        } else if (arg == "weight-count") {
            args.enable_weight_count = true;
        } else if (arg == "degree-extremes") {
            args.enable_degree_extremes = true;
        } else if (arg == "degree-histogram-width") {
            args.enable_degree_histogram_width = true;
        } else if (arg == "degree-histogram-count") {
            args.enable_degree_histogram_count = true;
        } else if (arg == "arc-length-average") {
            args.enable_arc_length_average = true;
        } else if (arg == "arc-length-histogram-width") {
            args.enable_arc_length_histogram_width = true;
        } else if (arg == "arc-length-histogram-count") {
            args.enable_arc_length_histogram_count = true;
        } else if (arg == "pair-distances-average") {
            args.enable_pair_distances_average = true;
        } else if (arg == "pair-distances-histogram-width") {
            args.enable_pair_distances_histogram_width = true;
        } else if (arg == "pair-distances-histogram-count") {
            args.enable_pair_distances_histogram_count = true;
        } else if (arg == "avg-apsp") {
            args.enable_all_pairs_shortest_path = true;
        } else if (arg == "avg-apsp-inv") {
            args.enable_all_pairs_shortest_path_inverse = true;
        } else if (arg == "avg-cluster") {
            args.enable_clustering_coefficient = true;
        } else if (arg == "avg-centrality") {
            args.enable_betweenness_centrality_average = true;
        } else if (arg == "area-connec") {
            args.enable_area_connectivity = true;
        } else if (arg == "modularity") {
            args.enable_modularity = true;
        } else if (arg == "tri-motifs") {
            args.enable_network_motifs = true;
        } else if (arg == "assortativity") {
            args.enable_assortativity = true;
        } else if (arg == "reciprocity") {
            args.enable_reciprocity = true;
        } else if (arg == "rich-club") {
            args.enable_rich_club = true;
        } else if (arg == "transitivity") {
            args.enable_transitivity = true;
        } else if (arg == "dia-approx") {
            args.enable_diameter_approximation = true;
        } else if (arg == "scc") {
            args.enable_strongly_connected_components = true;
        } else if (arg == "max-flow") {
            args.enable_maximum_flow = true;
        } else {
            spdlog::error("The argument \"{}\" is not a valid algorithm", arg);
            // terminate the programm and return a failure exitcode
            return EXIT_FAILURE;
        }
    }

    if (apsp_algorithm_arg == "dijkstra") {
        args.apsp_algorithm = SsspAlgorithm::Dijkstra;
    } else if (apsp_algorithm_arg == "delta-stepping") {
        args.apsp_algorithm = SsspAlgorithm::DeltaStepping;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --apsp-algorithm", apsp_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (apsp_delta_epoch_arg == "push-brute-force") {
        args.apsp_delta_epoch = DeltaSteppingEpochType::PushBruteForce;
    } else if (apsp_delta_epoch_arg == "push-check") {
        args.apsp_delta_epoch = DeltaSteppingEpochType::PushCheck;
    } else if (apsp_delta_epoch_arg == "push-short-long") {
        args.apsp_delta_epoch = DeltaSteppingEpochType::PushShortLong;
    } else if (apsp_delta_epoch_arg == "push-inner-short") {
        args.apsp_delta_epoch = DeltaSteppingEpochType::PushInnerShort;
    } else if (apsp_delta_epoch_arg == "pull-model") {
        args.apsp_delta_epoch = DeltaSteppingEpochType::PullModel;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --apsp-delta-epoch", apsp_delta_epoch_arg);
        return EXIT_FAILURE;
    }

    if (dijkstra_queue_arg == "priority-queue") {
        args.dijkstra_queue = DijkstraQueueType::PriorityQueue;
    } else if (dijkstra_queue_arg == "fibonacci-heap") {
        args.dijkstra_queue = DijkstraQueueType::FibonacciHeap;
    } else if (dijkstra_queue_arg == "bucket-queue") {
        args.dijkstra_queue = DijkstraQueueType::BucketQueue;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --dijkstra-queue", dijkstra_queue_arg);
        return EXIT_FAILURE;
    }

    if (centrality_algorithm_arg == "brandes") {
        args.centrality_algorithm = BetweennessCentralityAlgorithm::Brandes;
    } else if (centrality_algorithm_arg == "path-enumeration") {
        args.centrality_algorithm = BetweennessCentralityAlgorithm::PathEnumeration;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --centrality-algorithm", centrality_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (modularity_algorithm_arg == "louvain") {
        args.modularity_algorithm = ModularityAlgorithm::Louvain;
    } else if (modularity_algorithm_arg == "leiden") {
        args.modularity_algorithm = ModularityAlgorithm::Leiden;
    } else if (modularity_algorithm_arg == "label-propagation") {
        args.modularity_algorithm = ModularityAlgorithm::LabelPropagation;
    } else if (modularity_algorithm_arg == "infomap") {
        args.modularity_algorithm = ModularityAlgorithm::Infomap;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --modularity-algorithm", modularity_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (args.enable_all_pairs_shortest_path_inverse && args.dijkstra_queue == DijkstraQueueType::BucketQueue) {
        spdlog::error("--dijkstra-queue bucket-queue cannot be used with avg-apsp-inv: the inverse distances are not integers");
        return EXIT_FAILURE;
    }

    if (network_motif_algorithm_arg == "questions") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::Questions;
    } else if (network_motif_algorithm_arg == "rma") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::Rma;
    } else if (network_motif_algorithm_arg == "rma-accumulate") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::RmaAccumulate;
    } else if (network_motif_algorithm_arg == "rma-fair") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::RmaFair;
    } else if (network_motif_algorithm_arg == "rma-fewer-arcs") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::RmaFewerArcs;
    } else if (network_motif_algorithm_arg == "rma-fewer-messages") {
        args.network_motif_algorithm = NetworkMotifAlgorithm::RmaFewerMessages;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --tri-motifs-algorithm", network_motif_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (scc_algorithm_arg == "tarjan-gathered") {
        args.scc_algorithm = SccAlgorithm::TarjanGathered;
    } else if (scc_algorithm_arg == "forward-backward") {
        args.scc_algorithm = SccAlgorithm::ForwardBackward;
    } else if (scc_algorithm_arg == "coloring") {
        args.scc_algorithm = SccAlgorithm::Coloring;
    } else if (scc_algorithm_arg == "multistep") {
        args.scc_algorithm = SccAlgorithm::Multistep;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --scc-algorithm", scc_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (max_flow_algorithm_arg == "ford-fulkerson") {
        args.max_flow_algorithm = MaxFlowAlgorithm::FordFulkerson;
    } else if (max_flow_algorithm_arg == "edmonds-karp") {
        args.max_flow_algorithm = MaxFlowAlgorithm::EdmondsKarp;
    } else if (max_flow_algorithm_arg == "dinic") {
        args.max_flow_algorithm = MaxFlowAlgorithm::Dinic;
    } else if (max_flow_algorithm_arg == "orlin-king-rao-tarjan") {
        args.max_flow_algorithm = MaxFlowAlgorithm::OrlinKingRaoTarjan;
    } else if (max_flow_algorithm_arg == "distributed-push-relabel") {
        args.max_flow_algorithm = MaxFlowAlgorithm::DistributedPushRelabel;
    } else {
        spdlog::error("The argument \"{}\" is not a valid --max-flow-algorithm", max_flow_algorithm_arg);
        return EXIT_FAILURE;
    }

    if (args.enable_maximum_flow) {
        if (max_flow_sources_arg.empty() || max_flow_sinks_arg.empty()) {
            spdlog::error("--max-flow-sources and --max-flow-sinks must each list at least one rank:node_id pair when max-flow is selected");
            return EXIT_FAILURE;
        }

        for (const auto& entry : max_flow_sources_arg) {
            const auto node = parse_node_identifier(entry);
            if (!node) {
                spdlog::error("The argument \"{}\" in --max-flow-sources is not a valid rank:node_id pair", entry);
                return EXIT_FAILURE;
            }
            args.max_flow_sources.push_back(*node);
        }

        for (const auto& entry : max_flow_sinks_arg) {
            const auto node = parse_node_identifier(entry);
            if (!node) {
                spdlog::error("The argument \"{}\" in --max-flow-sinks is not a valid rank:node_id pair", entry);
                return EXIT_FAILURE;
            }
            args.max_flow_sinks.push_back(*node);
        }
    }

    args.undirected = static_cast<bool>(*flag_undirected);
    args.one_weight = static_cast<bool>(*flag_one_weight);
    args.remove_autapses = static_cast<bool>(*flag_remove_autapses);
    args.output_file = args.input_directory / "metrics";

    if (!output_directory.empty()) {
        args.output_file /= output_directory;
    }

    mpiPP::MPIProgress::set_disable_status(static_cast<bool>(*flag_disable_status));

    const auto disable_timer_report = static_cast<bool>(*flag_disable_timer);
    utility::RegionTimer::set_enabled(!disable_timer_report);
    mpiPP::MPIRegionTimerReport::set_disable_status(disable_timer_report);

    mpiPP::MPIWrapper::init(argument_count, arguments);

    print_arguments(argument_count, arguments);
    calculate_metrics(args);

    mpiPP::MPIWrapper::finalize();

    return 0;
}
