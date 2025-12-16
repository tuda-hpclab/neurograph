#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Histogram.h"
#include "Types.h"

#include <array>
#include <filesystem>
#include <optional>
#include <tuple>
#include <vector>

class MetricStore {
public:
    void set_global_number_nodes(const global_node_id_type number) {
        number_nodes = number;
    }

    void set_number_in_arcs(const arc_id_type number) {
        number_in_arcs = number;
    }

    void set_number_out_arcs(const arc_id_type number) {
        number_out_arcs = number;
    }

    void set_weight_in_arcs(const weight_type weight) {
        weight_in_arcs = weight;
    }

    void set_weight_out_arcs(const weight_type weight) {
        weight_out_arcs = weight;
    }

    void set_extreme_in_degrees(const MinMax min_max) {
        extreme_in_degrees = min_max;
    }

    void set_in_degree_histogram_width(Histogram<arc_id_type> hist) {
        in_degree_histogram_width = std::move(hist);
    }

    void set_in_degree_histogram_count(Histogram<arc_id_type> hist) {
        in_degree_histogram_count = std::move(hist);
    }

    void set_extreme_out_degrees(const MinMax min_max) {
        extreme_out_degrees = min_max;
    }

    void set_out_degree_histogram_width(Histogram<arc_id_type> hist) {
        out_degree_histogram_width = std::move(hist);
    }

    void set_out_degree_histogram_count(Histogram<arc_id_type> hist) {
        out_degree_histogram_count = std::move(hist);
    }

    void set_average_pair_distances(const double length) {
        average_pair_distance = length;
    }

    void set_pair_distances_histogram_width(Histogram<double> hist) {
        pair_distances_histogram_width = std::move(hist);
    }

    void set_pair_distances_histogram_count(Histogram<double> hist) {
        pair_distances_histogram_count = std::move(hist);
    }

    void set_average_arc_length(const double length) {
        average_arc_length = length;
    }

    void set_arc_length_histogram_width(Histogram<double> hist) {
        arc_length_histogram_width = std::move(hist);
    }

    void set_arc_length_histogram_count(Histogram<double> hist) {
        arc_length_histogram_count = std::move(hist);
    }

    void set_all_pairs_shortest_paths_result(apsp_global_result<distance_type> result) {
        apsp_result = std::move(result);
    }

    void set_all_pairs_shortest_paths_inverse_result(apsp_global_result<inverse_distance_type> result) {
        apsp_inverse_result = std::move(result);
    }

    void set_average_clustering_coefficient(const double coefficient) {
        average_clustering_coefficient = coefficient;
    }

    void set_average_clustering_coefficient_2(const double coefficient) {
        average_clustering_coefficient_2 = coefficient;
    }

    void set_average_betweenness_centrality(const double centrality) {
        average_betweenness_centrality = centrality;
    }

    void set_area_connectivity_map(AreaConnectivityMap map) {
        area_connectivity_strength = std::move(map);
    }

    void set_motifs(const std::array<long double, 14>& motif_counts) {
        motifs = motif_counts;
    }

    void set_assortativity(const std::tuple<double, double, double, double>& coefficients) {
        assortativity = coefficients;
    }

    void set_approximate_diameter(const distance_type diameter) {
        approximate_diameter = diameter;
    }

    void output(const std::filesystem::path& output_path) const;

private:
    std::optional<global_node_id_type> number_nodes{};
    std::optional<arc_id_type> number_in_arcs{};
    std::optional<arc_id_type> number_out_arcs{};
    std::optional<weight_type> weight_in_arcs{};
    std::optional<weight_type> weight_out_arcs{};

    std::optional<MinMax> extreme_in_degrees{};
    std::optional<Histogram<arc_id_type>> in_degree_histogram_width{};
    std::optional<Histogram<arc_id_type>> in_degree_histogram_count{};

    std::optional<MinMax> extreme_out_degrees{};
    std::optional<Histogram<arc_id_type>> out_degree_histogram_width{};
    std::optional<Histogram<arc_id_type>> out_degree_histogram_count{};

    std::optional<double> average_pair_distance{};
    std::optional<Histogram<double>> pair_distances_histogram_width{};
    std::optional<Histogram<double>> pair_distances_histogram_count{};

    std::optional<double> average_arc_length{};
    std::optional<Histogram<double>> arc_length_histogram_width{};
    std::optional<Histogram<double>> arc_length_histogram_count{};

    std::optional<apsp_global_result<distance_type>> apsp_result{};
    std::optional<apsp_global_result<inverse_distance_type>> apsp_inverse_result{};

    std::optional<double> average_clustering_coefficient{};
    std::optional<double> average_clustering_coefficient_2{};

    std::optional<double> average_betweenness_centrality{};

    std::optional<AreaConnectivityMap> area_connectivity_strength{};

    std::optional<std::array<long double, 14>> motifs{};

    std::optional<std::tuple<double, double, double, double>> assortativity{};

    std::optional<distance_type> approximate_diameter{};
};
