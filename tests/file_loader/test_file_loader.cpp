/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_file_loader.h"

#include "Types.h"

#include "graph/FileLoader.h"
#include "graph/GraphTypes.h"
#include "utility/Vec3.h"

#include <cpp-utility/Exception.hpp>

#include <filesystem>
#include <fstream>
#include <span>
#include <utility>
#include <vector>

TEST_F(FileLoaderTest, loadNodesHandlesCommentsBlanksMalformedAndMissingArea) {
    // All tests write their own files and therefore only run with a single rank to avoid concurrent writes to the same path.
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./file_loader_input");
    const auto path = std::filesystem::path{ "./file_loader_input/positions.txt" };

    {
        auto file = std::ofstream{ path };
        file << "# a comment line\n";                 // comment -> skipped
        file << "\n";                                 // blank line -> skipped
        file << "not a valid node line\n";            // parse failure -> skipped
        file << "1 0.1 0.2 0.3 excitatory\n";         // no area name -> defaults to "unknown"
        file << "5 0.4 0.5 0.6 area_1 inhibitory\n";  // wrong id (expected 2) -> skipped
        file << "2 0.7 0.8 0.9 area_1 excitatory\n";  // valid, with area name
    }

    const auto nodes = FileLoader::load_nodes(path, mpi_rank_type{ 0 });

    ASSERT_EQ(nodes.positions.size(), 2);
    ASSERT_EQ(nodes.positions[0], Vec3d(0.1, 0.2, 0.3));
    ASSERT_EQ(nodes.positions[1], Vec3d(0.7, 0.8, 0.9));

    ASSERT_EQ(nodes.area_names_ind.size(), 2);
    ASSERT_EQ(nodes.signal_types_ind.size(), 2);

    // the first node has no area name in the file, so it defaults to "unknown"
    ASSERT_EQ(nodes.area_names[nodes.area_names_ind[0]], "unknown");
    ASSERT_EQ(nodes.area_names[nodes.area_names_ind[1]], "area_1");

    ASSERT_EQ(nodes.signal_types[nodes.signal_types_ind[0]], "excitatory");
    ASSERT_EQ(nodes.signal_types[nodes.signal_types_ind[1]], "excitatory");
}

TEST_F(FileLoaderTest, loadInArcsSkipsInvalidLinesAndKeepsValidArcs) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./file_loader_input");
    const auto path = std::filesystem::path{ "./file_loader_input/in_network.txt" };

    {
        auto file = std::ofstream{ path };
        file << "# comment\n";     // comment -> skipped
        file << "\n";              // blank -> skipped
        file << "garbage line\n";  // parse failure -> skipped
        file << "0 1 0 2 1 0\n";   // plastic flag 0 -> skipped
        file << "0 1 0 2 0 1\n";   // weight 0 -> skipped
        file << "1 1 0 2 1 1\n";   // target rank != my rank -> skipped
        file << "0 99 0 2 1 1\n";  // target id out of range -> skipped
        file << "0 1 5 1 1 1\n";   // source rank out of range -> skipped
        file << "0 1 0 99 1 1\n";  // source id out of range -> skipped
        file << "0 1 0 2 4 1\n";   // valid: node 0 <- node 1 with weight 4
    }

    const auto node_distribution = std::vector<node_id_type>{ 3 };
    const auto in_arcs = FileLoader::load_in_arcs(path, mpi_rank_type{ 0 }, node_distribution);

    ASSERT_EQ(in_arcs.size(), 3);
    ASSERT_EQ(in_arcs[0].size(), 1);
    ASSERT_EQ(in_arcs[0].at(std::pair{ mpi_rank_type{ 0 }, node_id_type{ 1 } }), weight_type{ 4 });
    ASSERT_TRUE(in_arcs[1].empty());
    ASSERT_TRUE(in_arcs[2].empty());
}

TEST_F(FileLoaderTest, loadOutArcsSkipsInvalidLinesAndKeepsValidArcs) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./file_loader_input");
    const auto path = std::filesystem::path{ "./file_loader_input/out_network.txt" };

    {
        auto file = std::ofstream{ path };
        file << "# comment\n";     // comment -> skipped
        file << "\n";              // blank -> skipped
        file << "garbage line\n";  // parse failure -> skipped
        file << "0 1 0 1 1 0\n";   // plastic flag 0 -> skipped
        file << "0 1 0 1 0 1\n";   // weight 0 -> skipped
        file << "0 1 5 1 1 1\n";   // source rank != my rank -> skipped
        file << "0 1 0 99 1 1\n";  // source id out of range -> skipped
        file << "5 1 0 1 1 1\n";   // target rank out of range -> skipped
        file << "0 99 0 1 1 1\n";  // target id out of range -> skipped
        file << "0 2 0 1 7 1\n";   // valid: node 0 -> node 1 with weight 7
    }

    const auto node_distribution = std::vector<node_id_type>{ 3 };
    const auto out_arcs = FileLoader::load_out_arcs(path, mpi_rank_type{ 0 }, node_distribution);

    ASSERT_EQ(out_arcs.size(), 3);
    ASSERT_EQ(out_arcs[0].size(), 1);
    ASSERT_EQ(out_arcs[0].at(std::pair{ mpi_rank_type{ 0 }, node_id_type{ 1 } }), weight_type{ 7 });
    ASSERT_TRUE(out_arcs[1].empty());
    ASSERT_TRUE(out_arcs[2].empty());
}

TEST_F(FileLoaderTest, loadInArcsThrowsWhenFileCannotBeOpened) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./file_loader_input");

    const auto node_distribution = std::vector<node_id_type>{ 3 };
    ASSERT_THROW(static_cast<void>(FileLoader::load_in_arcs("./file_loader_input/does_not_exist_in.txt", mpi_rank_type{ 0 }, node_distribution)),
                 utility::Exception);
}

TEST_F(FileLoaderTest, loadOutArcsThrowsWhenFileCannotBeOpened) {
    if (skip_unless_rank_count(1)) {
        return;
    }

    std::filesystem::create_directories("./file_loader_input");

    const auto node_distribution = std::vector<node_id_type>{ 3 };
    ASSERT_THROW(static_cast<void>(FileLoader::load_out_arcs("./file_loader_input/does_not_exist_out.txt", mpi_rank_type{ 0 }, node_distribution)),
                 utility::Exception);
}
