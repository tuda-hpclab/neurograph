# Formatting

Two conventions hold for every file under `source/` and `tests/`: the copyright
header each one starts with, and the order of its includes.

## Copyright header

Every source file starts with the header below, optionally preceded by
`#pragma once` and exactly one empty line:

```c++
/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */
```

Accepted as the year is the current one, or an ascending interval `XXXX-YYYY`
whose second year is the current one.

## Include order

Only the headers of this project are included in `""`; every dependency,
cpp-utility and mpi-wrapper included, is included in `<>`. The includes at the
top of a file form these blocks, in this order, separated by exactly one empty
line and sorted by byte order within a block:

| Block | Contents |
| --- | --- |
| 1 | the header the source file belongs to, in `""` and without a directory |
| 2 | the headers next to this one, in `""` and without a directory, in particular `"GraphTest.h"` |
| 3 | `"Types.h"`, the project wide type aliases |
| 4 | the other headers of this project, in `""` and with a directory |
| 5 | everything of CLI11, in `<>` |
| 6 | everything of cpp-utility, in `<>` |
| 7 | everything of {fmt}, in `<>` |
| 8 | everything of GoogleTest, in `<>` |
| 9 | `<mpi.h>` |
| 10 | everything of mpi-wrapper, in `<>` |
| 11 | `<omp.h>` |
| 12 | everything of range-v3, in `<>` |
| 13 | everything of spdlog, in `<>` |
| 14 | the standard headers, in `<>` |

Every block is optional, the ones that are there have to be in this order, and
blocks 5 to 13 are ordered by the name of the library.
