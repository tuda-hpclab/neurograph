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

#include <random>

/**
 * @brief A source of uniformly distributed random numbers from [0, 1).
 *		Randomized algorithms draw their random numbers through this interface,
 *		so callers (especially tests) can inject their own deterministic generator.
 */
class RandomNumberGenerator {
public:
    RandomNumberGenerator() = default;
    RandomNumberGenerator(const RandomNumberGenerator&) = delete;
    RandomNumberGenerator(RandomNumberGenerator&&) = delete;
    RandomNumberGenerator& operator=(const RandomNumberGenerator&) = delete;
    RandomNumberGenerator& operator=(RandomNumberGenerator&&) = delete;
    virtual ~RandomNumberGenerator() = default;

    /**
     * @brief Draws the next random number.
     * @return A uniformly distributed random number from [0, 1)
     */
    [[nodiscard]] virtual double draw() = 0;
};

/**
 * @brief The generator that is used whenever the caller does not supply their own:
 *		a std::mt19937 behind a uniform real distribution on [0, 1).
 */
class DefaultRandomNumberGenerator : public RandomNumberGenerator {
public:
    /**
     * @brief Constructs the generator.
     * @param seed The seed for the underlying std::mt19937
     */
    explicit DefaultRandomNumberGenerator(const unsigned int seed)
        : prng(seed) { }

    [[nodiscard]] double draw() override {
        return distribution(prng);
    }

private:
    std::mt19937 prng;
    std::uniform_real_distribution<double> distribution{ 0.0, 1.0 };
};
