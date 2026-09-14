/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_motif_structure.h"

#include "metrics/local_structure/NetworkMotifs.h"

#include <cstdint>

// The algorithms of NetworkMotifs start with a triple that can still be any motif and narrow it
// down by unsetting the motifs its arcs rule out, so the bit array is checked here on its own:
// which bit a motif number occupies, that setting and unsetting are inverse, and that
// check_validity accepts exactly the states a classified triple can end up in.

namespace {
constexpr auto first_motif_type = 1U;
constexpr auto number_motif_types = 14U;
} // namespace

TEST_F(ThreeMotifStructureTest, testSetSingleMotifType) {
    for (auto motif_type = first_motif_type; motif_type < number_motif_types; ++motif_type) {
        auto motif_structure = NetworkMotifs::ThreeMotifStructure{};

        motif_structure.set_motif_types({ motif_type });

        // The motif number is the index of its bit, so index 0 stays unused
        const auto expected_bit_array = static_cast<std::uint16_t>(1U << motif_type);
        EXPECT_EQ(motif_structure.motif_type_bit_array, expected_bit_array) << "motif " << motif_type;

        for (auto other_type = first_motif_type; other_type < number_motif_types; ++other_type) {
            EXPECT_EQ(motif_structure.is_motif_type_set(other_type), other_type == motif_type) << "motif " << other_type << " after setting " << motif_type;
        }
    }
}

TEST_F(ThreeMotifStructureTest, testSetAndUnsetSeveralMotifTypes) {
    auto motif_structure = NetworkMotifs::ThreeMotifStructure{};

    motif_structure.set_motif_types({ 1, 5, 13 });

    EXPECT_TRUE(motif_structure.is_motif_type_set(1));
    EXPECT_TRUE(motif_structure.is_motif_type_set(5));
    EXPECT_TRUE(motif_structure.is_motif_type_set(13));
    EXPECT_FALSE(motif_structure.is_motif_type_set(4));

    // Setting a motif that is already set does not change anything, and unsetting one leaves the others alone
    motif_structure.set_motif_types({ 5 });
    motif_structure.unset_motif_types({ 5 });

    EXPECT_TRUE(motif_structure.is_motif_type_set(1));
    EXPECT_FALSE(motif_structure.is_motif_type_set(5));
    EXPECT_TRUE(motif_structure.is_motif_type_set(13));

    motif_structure.unset_motif_types({ 1, 13 });

    EXPECT_EQ(motif_structure.motif_type_bit_array, 0);
}

TEST_F(ThreeMotifStructureTest, testSelfTest) {
    auto motif_structure = NetworkMotifs::ThreeMotifStructure{};

    // Sets and unsets every motif, once one at a time and once all of them at once; it asserts
    // internally and has to leave the bit array empty
    motif_structure.self_test();

    EXPECT_EQ(motif_structure.motif_type_bit_array, 0);
}

TEST_F(ThreeMotifStructureTest, testCheckValidity) {
    auto motif_structure = NetworkMotifs::ThreeMotifStructure{};

    // A triple that matched no case of the classification keeps its empty bit array
    EXPECT_TRUE(motif_structure.check_validity());

    // A classified triple is of exactly one motif
    motif_structure.set_motif_types({ 7 });
    EXPECT_TRUE(motif_structure.check_validity());

    // Two motifs at once cannot happen; the rejected bit array is printed to stdout by check_validity
    motif_structure.set_motif_types({ 9 });
    EXPECT_FALSE(motif_structure.check_validity());

    motif_structure.unset_motif_types({ 7 });
    EXPECT_TRUE(motif_structure.check_validity());
}
