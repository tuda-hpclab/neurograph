#include "graph/DistributedGraph.h"
#include "metrics/AllPairsDistances.h"
#include "metrics/AllPairsDistancesHistogram.h"
#include "metrics/AllPairsShortestPath.h"
#include "metrics/ArcCounter.h"
#include "metrics/ArcLength.h"
#include "metrics/ArcLengthHistogram.h"
#include "metrics/AreaConnectivity.h"
#include "metrics/Assortativity.h"
#include "metrics/Centrality.h"
#include "metrics/Clustering.h"
#include "metrics/Degree.h"
#include "metrics/DegreeHistogram.h"
#include "metrics/DiameterApproximation.h"
#include "metrics/NetworkMotifs.h"
#include "metrics/NodeCounter.h"
#include "metrics/WeightCounter.h"
#include "utility/MetricStore.h"
#include "utility/Status.h"
#include "utility/Timer.h"

#include "cpp-utility/Cast.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPISynchronization.h"
#include "mpi-wrapper/MPIWrapper.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    bool enable_all_pairs_shortest_path_inverse;
    bool enable_clustering_coefficient;
    bool enable_betweenness_centrality_average;
    bool enable_area_connectivity;
    bool enable_network_motifs;
    bool enable_assortativity;
    bool enable_diameter_approximation;

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
        const auto in_extremes = Degree::compute_extreme_in_degrees(dg);
        const auto out_extremes = Degree::compute_extreme_out_degrees(dg);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_extreme_in_degrees(in_extremes);
            ms.set_extreme_out_degrees(out_extremes);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_degree_histogram_width) {
        const auto width = args.bin_width_degree_histogram;
        auto in_degree_hist = DegreeHistogram::compute_in_degree_fixed_bin_width(dg, width);
        auto out_degree_hist = DegreeHistogram::compute_out_degree_fixed_bin_width(dg, width);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_in_degree_histogram_width(std::move(in_degree_hist));
            ms.set_out_degree_histogram_width(std::move(out_degree_hist));
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_degree_histogram_count) {
        const auto max_in_degree = Degree::compute_maximum_in_degree(dg);
        const auto max_out_degree = Degree::compute_maximum_out_degree(dg);
        auto in_degree_hist = DegreeHistogram::compute_in_degree_fixed_bin_count(dg, 0, max_in_degree + 1, max_in_degree + 1);
        auto out_degree_hist = DegreeHistogram::compute_out_degree_fixed_bin_count(dg, 0, max_out_degree + 1, max_out_degree + 1);

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
        const auto& apsp_result = AllPairsShortestPath::compute_apsp(dg);
        dg.unlock_all_rma_windows();

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_all_pairs_shortest_paths_result(apsp_result);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (args.enable_all_pairs_shortest_path_inverse) {
        dg.lock_all_rma_windows();
        const auto& apsp_result = AllPairsShortestPath::compute_apsp_inverse(dg);
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
        const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg);

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

    if (args.enable_network_motifs) {
        const auto& motifs = NetworkMotifs::compute_network_triple_motifs(dg);

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

    if (args.enable_diameter_approximation) {
        const auto approximate_diameter = DiameterApproximation::compute_approximation(dg, 1);

        if (mpiPP::MPIInfo::is_root_rank()) {
            ms.set_approximate_diameter(approximate_diameter);
        }
        mpiPP::MPISynchronization::barrier();
    }

    if (mpiPP::MPIInfo::is_root_rank()) {
        ms.output(args.output_file);
    }
}

static void print_arguments(const int argument_count, char* arguments[]) {
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    fmt::print("Arguments are:\n{}\n\n", fmt::join(std::span<char*>{ arguments, utility::save_cast<std::size_t>(argument_count) }, " "));
}

int main(const int argument_count, char* arguments[]) {
    auto app = CLI::App{ "" };
    auto args = CLIArguments{};

    auto* opt_input_directory = app.add_option("--input", args.input_directory, "The directory that contains the input files.")->required();
    opt_input_directory->check(CLI::ExistingDirectory);

    auto selected_algorithms = std::vector<std::string>{};

    app.add_flag("--func", selected_algorithms,
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
                 "tri-motifs,\n"
                 "assortativity,\n"
                 "dia-approx")
        ->delimiter(',')
        ->allow_extra_args(true);

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

    auto output_directory = std::string{ "" };
    app.add_option("--output-directory", output_directory);

    auto* flag_disable_status = app.add_flag("--disable-status-report", "Disables reporting of the status");
    auto* flag_disable_timer = app.add_flag("--disable-timer-report", "Disables reporting of the timer");

    CLI11_PARSE(app, argument_count, arguments);

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
        } else if (arg == "tri-motifs") {
            args.enable_network_motifs = true;
        } else if (arg == "assortativity") {
            args.enable_assortativity = true;
        } else if (arg == "dia-approx") {
            args.enable_diameter_approximation = true;
        } else {
            spdlog::error("The argument \"{}\" is not a valid algorithm", arg);
            // terminate the programm and return a failure exitcode
            return EXIT_FAILURE;
        }
    }

    args.undirected = static_cast<bool>(*flag_undirected);
    args.one_weight = static_cast<bool>(*flag_one_weight);
    args.remove_autapses = static_cast<bool>(*flag_remove_autapses);
    args.output_file = args.input_directory / "metrics";

    if (!output_directory.empty()) {
        args.output_file /= output_directory;
    }

    Status::set_disable_status(static_cast<bool>(*flag_disable_status));
    Timer::set_disable_status(static_cast<bool>(*flag_disable_timer));

    mpiPP::MPIWrapper::init(argument_count, arguments);

    print_arguments(argument_count, arguments);
    calculate_metrics(args);

    mpiPP::MPIWrapper::finalize();

    return 0;
}
