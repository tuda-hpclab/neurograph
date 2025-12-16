/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MetricStore.h"

#include "mpi-wrapper/MPIInfo.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/std.h>

#include <fstream>
#include <iostream>
#include <sstream>

void MetricStore::output(const std::filesystem::path& output_path) const {
    if (!mpiPP::MPIInfo::is_root_rank()) {
        return;
    }

    if (std::filesystem::is_regular_file(output_path)) {
        std::cerr << "Output path is a file: " << output_path << std::endl;
        return;
    }

    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path);
    }

    const auto output_node_count = [this, output_path]() {
        if (!number_nodes) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The total number of nodes in the graph is: {}\n", number_nodes.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "node_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_arc_count = [this, output_path]() {
        if (!number_out_arcs || !number_in_arcs) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The total number of in arcs in the graph is: {}\n", number_in_arcs.value());
        fmt::print(ss, "The total number of out arcs in the graph is: {}\n", number_out_arcs.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "arc_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_arc_weight = [this, output_path]() {
        if (!weight_out_arcs || !weight_in_arcs) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The total weight of in arcs in the graph is: {}\n", weight_in_arcs.value());
        fmt::print(ss, "The total weight of out arcs in the graph is: {}\n", weight_out_arcs.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "weight_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_extreme_degrees = [this, output_path]() {
        if (!extreme_in_degrees || !extreme_out_degrees) {
            return;
        }

        const auto [min_in, max_in] = extreme_in_degrees.value();
        const auto [min_out, max_out] = extreme_out_degrees.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The minimum in-degree is: {}\nThe maximum in-degree is: {}\n", min_in, max_in);
        fmt::print(ss, "The minimum out-degree is: {}\nThe maximum out-degree is: {}\n", min_out, max_out);

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "extreme_degrees.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_degree_histogram_width = [this, output_path]() {
        if (!in_degree_histogram_width || !out_degree_histogram_width) {
            return;
        }

        const auto& in_degree_hist = in_degree_histogram_width.value();
        const auto& out_degree_hist = out_degree_histogram_width.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "Histogram of in degrees:\n");
        ss << in_degree_hist;

        fmt::print(ss, "Histogram of out degrees:\n");
        ss << out_degree_hist;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "degree_histograms_width.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_degree_histograms_count = [this, output_path]() {
        if (!in_degree_histogram_count || !out_degree_histogram_count) {
            return;
        }

        const auto& in_degree_hist = in_degree_histogram_count.value();
        const auto& out_degree_hist = out_degree_histogram_count.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "Histogram of in degrees:\n");
        ss << in_degree_hist;

        fmt::print(ss, "Histogram of out degrees:\n");
        ss << out_degree_hist;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "degree_histograms_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_average_pair_distances = [this, output_path]() {
        if (!average_pair_distance) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The average pair distance is: {}\n", average_pair_distance.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "average_pair_distance.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_pair_distances_histogram_width = [this, output_path]() {
        if (!pair_distances_histogram_width) {
            return;
        }

        const auto& distances = pair_distances_histogram_width.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The pair-distances histogram is given by:\n");

        ss << distances;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "pair_distances_histogram_width.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_pair_distances_histogram_count = [this, output_path]() {
        if (!pair_distances_histogram_count) {
            return;
        }

        const auto& distances = pair_distances_histogram_count.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The pair-distances histogram is given by:\n");

        ss << distances;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "pair_distances_histogram_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_average_arc_length = [this, output_path]() {
        if (!average_arc_length) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The average arc length is: {}\n", average_arc_length.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "average_arc_length.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_arc_length_histogram_width = [this, output_path]() {
        if (!arc_length_histogram_width) {
            return;
        }

        const auto& hist = arc_length_histogram_width.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The arc-length histogram is given by:\n");
        ss << hist;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "arc_length_histogram_width.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_arc_length_histogram_count = [this, output_path]() {
        if (!arc_length_histogram_count) {
            return;
        }

        const auto& hist = arc_length_histogram_count.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The arc-length histogram is given by:\n");
        ss << hist;

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "arc_length_histogram_count.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_all_pairs_shortest_paths = [this, output_path]() {
        if (!apsp_result) {
            return;
        }

        const auto& [average_shortest_path, global_efficiency, diameter, number_unreachables, cluster_sizes] = apsp_result.value();

        auto ss = std::stringstream{};
        for (const auto& [cluster_size, occurence] : cluster_sizes) {
            fmt::print(ss, "Size: {}\tOccurences: {}\n", cluster_size, occurence);
        }

        fmt::print(ss, "The average shortest path is: {}\n", average_shortest_path);
        fmt::print(ss, "The global efficiency is: {}\n", global_efficiency);
        fmt::print(ss, "The diameter of the graph is: {}\n", diameter);
        fmt::print(ss, "However, the number of pairs without path is: {}\n", number_unreachables);
        fmt::print(ss, "The clusters were distributed as follows:\n");

        for (const auto& [cluster_size, occurence] : cluster_sizes) {
            fmt::print(ss, "Size: {}\tOccurences: {}\n", cluster_size, occurence);
        }

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "all_pairs_shortest_paths.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_all_pairs_shortest_paths_inverse = [this, output_path]() {
        if (!apsp_inverse_result) {
            return;
        }

        const auto& [average_shortest_path, global_efficiency, diameter, number_unreachables, cluster_sizes] = apsp_inverse_result.value();

        auto ss = std::stringstream{};
        for (const auto& [cluster_size, occurence] : cluster_sizes) {
            fmt::print(ss, "Size: {}\tOccurences: {}\n", cluster_size, occurence);
        }

        fmt::print(ss, "The average shortest path is: {}\n", average_shortest_path);
        fmt::print(ss, "The global efficiency is: {}\n", global_efficiency);
        fmt::print(ss, "The diameter of the graph is: {}\n", diameter);
        fmt::print(ss, "However, the number of pairs without path is: {}\n", number_unreachables);
        fmt::print(ss, "The clusters were distributed as follows:\n");

        for (const auto& [cluster_size, occurence] : cluster_sizes) {
            fmt::print(ss, "Size: {}\tOccurences: {}\n", cluster_size, occurence);
        }

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "all_pairs_shortest_paths_inverse.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_clustering_coefficients = [this, output_path]() {
        if (!average_clustering_coefficient || !average_clustering_coefficient_2) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The average clustering coefficient (i->j->k->i) is: {}\n", average_clustering_coefficient.value());
        fmt::print(ss, "The average clustering coefficient (i->j->k<-i) is: {}\n", average_clustering_coefficient_2.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "average_clustering_coefficients.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_average_betweenness_centrality = [this, output_path]() {
        if (!average_betweenness_centrality) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The average betweenness centrality is: {}\n", average_betweenness_centrality.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "average_betweenness_centrality.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_area_connectivity = [this, output_path]() {
        if (!area_connectivity_strength) {
            return;
        }

        const auto& area_connectivity = area_connectivity_strength.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The area connectivity is given by:\n");
        auto nr = std::size_t{ 0 };
        for (const auto& [key, value] : area_connectivity) {
            fmt::print(ss, "Connection {}: weight = {} ({} --> {})\n", nr, value, key.first, key.second);
            nr++;
        }

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "area_connectivity.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_network_motifs = [this, output_path]() {
        if (!motifs) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The normalized network motif counts are:\n");

        for (auto i = std::size_t{ 1 }; i < motifs.value().size(); i++) {
            fmt::print(ss, "motif {} = {}\n", i, motifs.value()[i]);
        }

        fmt::print(ss, "The total count of tripple-motif occurrences is: {}\n", motifs.value()[0]);

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "motifs.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_assortativity = [this, output_path]() {
        if (!assortativity) {
            return;
        }

        const auto& [r_in_in, r_in_out, r_out_in, r_out_out] = assortativity.value();

        auto ss = std::stringstream{};
        fmt::print(ss, "The assortativity coefficient for in-in is: {}\n", r_in_in);
        fmt::print(ss, "The assortativity coefficient for in-out is: {}\n", r_in_out);
        fmt::print(ss, "The assortativity coefficient for out-in is: {}\n", r_out_in);
        fmt::print(ss, "The assortativity coefficient for out-out is: {}\n", r_out_out);

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "assortativity.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    const auto output_approximate_diameter = [this, output_path]() {
        if (!approximate_diameter) {
            return;
        }

        auto ss = std::stringstream{};
        fmt::print(ss, "The approximate diameter is: {}\n", approximate_diameter.value());

        const auto output = ss.str();
        std::cout << output;

        const auto output_file_path = output_path / "approximate_diameter.txt";
        auto output_file = std::ofstream{ output_file_path };
        output_file << output;
    };

    output_node_count();
    output_arc_count();
    output_arc_weight();
    output_extreme_degrees();
    output_degree_histogram_width();
    output_degree_histograms_count();
    output_average_pair_distances();
    output_pair_distances_histogram_width();
    output_pair_distances_histogram_count();
    output_average_arc_length();
    output_arc_length_histogram_width();
    output_arc_length_histogram_count();
    output_all_pairs_shortest_paths();
    output_all_pairs_shortest_paths_inverse();
    output_clustering_coefficients();
    output_average_betweenness_centrality();
    output_area_connectivity();
    output_network_motifs();
    output_assortativity();
    output_approximate_diameter();
}
