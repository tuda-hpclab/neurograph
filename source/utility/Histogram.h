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

#include "cpp-utility/Exception.hpp"

#include "mpi-wrapper/MPIReductions.h"

#include <climits>
#include <cstdint>
#include <exception>
#include <ostream>
#include <span>
#include <type_traits>
#include <vector>

/**
 * @brief A class that represents a histogram with fixed width bins.
 *		The histogram is represented by the borders of the bins and the counts of the data points in the bins.
 *		Can add the data points on the fly
 * @tparam DataType The data type to store, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class FixedWidthHistogram {
public:
    /**
     * @brief Constructs a new empty histogram with the given bin width
     * @param bin_width The bin width, must be positive
     * @exception std::invalid_argument Thrown if bin_width is not positive
     */
    FixedWidthHistogram(const DataType bin_width)
        : _bin_width(bin_width) {
        utility::Exception::check(bin_width > 0, "bin_width must be positive");
    }

    /**
     * @brief Adds a data point to the histogram and enlarges the histogram if necessary
     * @param data_point The data point to add, mus tbe non-negative
     * @exception std::invalid_argument Thrown if data_point is negative
     */
    void add_data_point(const DataType data_point) {
        utility::Exception::check(data_point >= 0, "data_point must be non-negative");

        const auto bin = static_cast<std::size_t>(data_point / _bin_width);
        ensure_large_enough(bin);
        ++counts[bin];
    }

    /**
     * @brief Returns the bin width of this histogram
     * @return The bin width
     */
    [[nodiscard]] DataType get_bin_width() const noexcept {
        return _bin_width;
    }

    /**
     * @brief Returns a span of the borders of the histogram.
     *		The size is the same as the counts -> The last border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *		get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i+1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Extracts the borders vector
     * @return The borders
     */
    [[nodiscard]] std::vector<DataType>&& get_borders_vector() && {
        return std::move(borders);
    }

    /**
     * @brief Extracts the counts vector
     * @return The borders
     */
    [[nodiscard]] std::vector<std::size_t>&& get_counts_vector() && {
        return std::move(counts);
    }

private:
    void ensure_large_enough(const std::size_t index) {
        const auto old_size = borders.size();
        if (index < old_size) {
            return;
        }

        if (borders.capacity() <= index + 1) {
            const auto new_size = index * 2 + 1;
            borders.reserve(new_size);
            counts.reserve(new_size);
        }

        borders.resize(index + 1);
        counts.resize(index + 1);

        for (auto i = old_size; i < index + 1; ++i) {
            const auto i_cast = static_cast<DataType>(i);
            borders[i] = i_cast * _bin_width;
        }
    }

    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};

    DataType _bin_width;
};

/**
 * @brief A class that represents a histogram with a fixed number of bins.
 * 		The histogram is represented by the borders of the bins and the counts of the data points in the bins.
 *		Can add the data points on the fly
 * @tparam DataType The data type to store, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class FixedSizeHistogram {
public:
    /**
     * @brief Constructs an empty histogram with a fixed number of bins
     * @param minimum The minimum value of the histogram
     * @param maximum Larger than the maximum value of the histogram
     * @param number_bins The number of bins in the histogram
     * @exception std::invalid_argument Thrown if maximum is not larger than minimum or number_bins is 0
     *		Also thrown if DataType is an integer type and if maximum - minimum is not divisible by number_bins
     */
    FixedSizeHistogram(const DataType minimum, const DataType maximum, const std::size_t number_bins)
        : _minimum(minimum)
        , _maximum(maximum) {
        utility::Exception::check(minimum < maximum, "maximum must be greater than minimum");
        utility::Exception::check(number_bins > 0, "number_bins must be positive");

        const auto distance = maximum - minimum;
        bin_width = distance / static_cast<DataType>(number_bins);

        if constexpr (std::is_integral_v<DataType>) {
            const auto remainder = utility::save_cast<std::size_t>(distance) % number_bins;
            utility::Exception::check(remainder == 0, "maximum - minimum must be divisible by number_bins");
        }

        counts.resize(number_bins, std::size_t(0));
        borders.reserve(number_bins);
        for (auto border = minimum; border < maximum; border += bin_width) {
            borders.emplace_back(border);
        }
    }

    /**
     * @brief Adds a data point into the histogram
     * @param data_point The data point, must be in [minimum, maximum)
     * @exception std::invalid_argument Thrown if data_point is not in [minimum, maximum)
     */
    void add_data_point(const DataType data_point) {
        utility::Exception::check(data_point >= _minimum, "data_point must be in the interval [minimum, maximum)");
        utility::Exception::check(data_point < _maximum, "data_point must be in the interval [minimum, maximum)");

        auto index = 0U;
        while (index + 1 < borders.size() && data_point >= borders[index + 1]) {
            index++;
        }

        ++counts[index];
    }

    /**
     * @brief Returns the bin width of this histogram
     * @return The bin width
     */
    [[nodiscard]] DataType get_bin_width() const noexcept {
        return bin_width;
    }

    /**
     * @brief Returns a span of the borders of the histogram.
     *		The size is the same as the counts -> The last border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *		get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i+1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Extracts the borders vector
     * @return The borders
     */
    [[nodiscard]] std::vector<DataType>&& get_borders_vector() && {
        return std::move(borders);
    }

    /**
     * @brief Extracts the counts vector
     * @return The borders
     */
    [[nodiscard]] std::vector<std::size_t>&& get_counts_vector() && {
        return std::move(counts);
    }

private:
    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};

    DataType bin_width;

    DataType _minimum{};
    DataType _maximum{};
};

/**
 * @brief This class represents a general histogram.
 *		It can be constructed from a FixedWidthHistogram or a FixedSizeHistogram.
 * 		It can sum the counts of the histogram on rank 0 or on all ranks.
 * @tparam DataType The data type to store, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class Histogram {
public:
    /**
     * @brief Constructs a histogram from a FixedWidthHistogram
     * @param histogram The FixedWidthHistogram to move from
     */
    Histogram(FixedWidthHistogram<DataType>&& histogram)
        : borders(std::move(histogram).get_borders_vector())
        , counts(std::move(histogram).get_counts_vector()) {
    }

    /**
     * @brief Constructs a histogram from a FixedSizeHistogram
     * @param histogram The FixedSizeHistogram to move from
     */
    Histogram(FixedSizeHistogram<DataType>&& histogram)
        : borders(std::move(histogram).get_borders_vector())
        , counts(std::move(histogram).get_counts_vector()) {
    }

    /**
     * @brief Returns a span of the borders of the histogram.
     *		The size is the same as the counts -> The last border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *		get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i+1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Reduces the histogram via summing the counts and changes the counts in each object.
     *		Must be called on all ranks. The result is on rank 0.
     *		Assumes that the borders are the same across all ranks.
     *	@exception std::logic_error Thrown if the histograms on different ranks have different sizes
     */
    void sum_on_rank_0() {
        const auto borders_size = borders.size();
        const auto counts_size = counts.size();

        const auto borders_max = mpiPP::MPIReductions::all_reduce_max(borders_size);
        const auto counts_max = mpiPP::MPIReductions::all_reduce_max(counts_size);

        borders.resize(borders_max, DataType(0));
        counts.resize(counts_max, std::size_t(0));

        auto summed_counts = mpiPP::MPIReductions::reduce_componentwise_sum(counts);
        counts = std::move(summed_counts);
    }

    /**
     * @brief Reduces the histogram via summing the counts and changes the counts in each object.
     *		Must be called on all ranks. The result is on every rank.
     *		Assumes that the borders are the same across all ranks.
     *	@exception std::logic_error Thrown if the histograms on different ranks have different sizes
     */
    void sum_on_all_ranks() {
        const auto borders_size = borders.size();
        const auto counts_size = counts.size();

        const auto borders_min = mpiPP::MPIReductions::all_reduce_min(borders_size);
        const auto borders_max = mpiPP::MPIReductions::all_reduce_max(borders_size);

        const auto counts_min = mpiPP::MPIReductions::all_reduce_min(counts_size);
        const auto counts_max = mpiPP::MPIReductions::all_reduce_max(counts_size);

        utility::Exception::check(borders_min == borders_max, "Histograms on different ranks have different sizes");
        utility::Exception::check(counts_min == counts_max, "Histograms on different ranks have different sizes");

        auto summed_counts = mpiPP::MPIReductions::all_reduce_componentwise_sum(counts);
        counts = std::move(summed_counts);
    }

    friend std::ostream& operator<<(std::ostream& stream, const Histogram& hist) {
        for (auto i = std::size_t(0); i < hist.borders.size(); ++i) {
            const auto start = hist.borders[i];
            const auto end = i + 1 < hist.borders.size() ? hist.borders[i + 1] : std::numeric_limits<DataType>::max();
            const auto count = hist.counts[i];

            stream << i << ". bin: " << start << '-' << end << ": " << count << '\n';
        }

        return stream;
    }

private:
    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};
};
