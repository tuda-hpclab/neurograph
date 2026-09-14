#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#if !defined(CPP_UTILITIES_NO_FMT)
#include <fmt/core.h>
#include <fmt/format.h>
#endif

#include <exception>
#include <string>
#include <utility>

namespace utility {

/**
 * Exception type with helpers for condition checks and runtime-formatted diagnostic messages.
 * Message formatting and storage can be disabled via Exception::hide_messages.
 *
 * Without {fmt}, i.e., with CPP_UTILITIES_NO_FMT defined, there is nothing that substitutes the
 * placeholders: the format string becomes the message as it is, placeholders included, and the
 * arguments are dropped.
 */
class Exception : public std::exception {
public:
    /**
     * @brief If true, failures skip formatting and throw an exception with an empty message.
     */
    static inline bool hide_messages{ false };

    /**
     * @brief Returns the cause of the exception, i.e., the stored message
     * @return A constant char pointer to the content of the message
     */
    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }

    /**
     * @brief Returns for a true condition; otherwise delegates to fail().
     * @tparam FormatString A string-like type
     * @tparam Args Different types that can be substituted into the placeholders
     * @param condition The condition to evaluate
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders, dropped without {fmt}
     * @throws fmt::format_error if formatting fails and messages are enabled. Never thrown without {fmt}.
     * @throws Exception if @p condition is false.
     */
    template <typename FormatString, typename... Args>
    static constexpr void check(const bool condition, FormatString&& format, Args&&... args) {
        if (condition) {
            return;
        }

        fail(std::forward<FormatString>(format), std::forward<Args>(args)...);
    }

    /**
     * @brief Formats and stores a diagnostic message, then throws an Exception.
     * @tparam FormatString A string-like type
     * @tparam Args Different types that can be substituted into the placeholders
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders, dropped without {fmt}
     * @throws fmt::format_error if formatting fails and messages are enabled. Never thrown without {fmt}.
     * @throws Exception after successful formatting, or immediately with an empty message when messages are hidden.
     */
    template <typename FormatString, typename... Args>
    [[noreturn]] static constexpr void fail(FormatString&& format, [[maybe_unused]] Args&&... args) {
        if (hide_messages) {
            throw Exception{};
        }

#if defined(CPP_UTILITIES_NO_FMT)
        throw Exception{ std::string{ std::forward<FormatString>(format) } };
#else
        auto message = fmt::format(fmt::runtime(std::forward<FormatString>(format)), std::forward<Args>(args)...);
        throw Exception{ std::move(message) };
#endif
    }

private:
    std::string message;

    /**
     * @brief Default constructs an instance with empty message
     */
    Exception() = default;

    /**
     * @brief Constructs an instance with the associated message
     * @param mes The message of the exception
     */
    explicit Exception(std::string&& mes)
        : message(std::move(mes)) {
    }
};

} // namespace utility
