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

#include <cpp-utility/data-structure/Vec3.hpp>

template<typename T>
using Vec3 = utility::Vec3<T>;

using Vec3f = utility::Vec3<float>;
using Vec3d = utility::Vec3<double>;
using Vec3s = utility::Vec3<std::size_t>;
