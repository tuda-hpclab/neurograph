# Testing

The test suite under [`tests/`](../tests/) uses GoogleTest. CTest registers the
`graph_tests` executable twice — via `mpiexec` with one and with four ranks —
so every metric is checked both single-process and distributed. The
four-rank tests have single-rank counterparts ("dummy" tests) that build the
same graph on one rank and check the same expected values; those expectations
come from independent reference implementations (e.g. bctpy for betweenness
centrality, a brute-force-verified Python reference for modularity).

See [building.md](building.md) for how to build and run the suite.

## Sanitizers and static analysis

The sanitizer and static analysis switches are regular CMake options, so each of
the configurations below is a plain build tree. The tests run under all of them
with one and with four ranks, so they also cover the MPI communication itself.

| Configuration | What it adds |
| --- | --- |
| `ENABLE_SANITIZER_ADDRESS` + `ENABLE_SANITIZER_UNDEFINED_BEHAVIOR` | ASan plus UBSan (including float and pointer checks); add `-fno-sanitize-recover` so a report fails the test |
| `ENABLE_SANITIZER_LEAK` | Standalone LeakSanitizer |
| `ENABLE_SANITIZER_THREAD` | ThreadSanitizer; build as `RelWithDebInfo`, because the gcov counters of a Debug build would be reported as data races |
| `_GLIBCXX_DEBUG`, `_GLIBCXX_DEBUG_PEDANTIC`, `_GLIBCXX_ASSERTIONS` | The debug mode of libstdc++ |
| `_FORTIFY_SOURCE=3`, stack protector, `relro`/`now` | Hardened release build |
| `ctest -T memcheck` | Valgrind with `--leak-check=full --track-origins=yes` |
| `ENABLE_SANITIZER_MEMORY` | MemorySanitizer; noisy, because the MPI library is not MSan-instrumented |
| `ENABLE_CLANG_TIDY`, `ENABLE_CPPCHECK` | Static analysis, compile step only |

For example, an ASan plus UBSan build:

```sh
cmake -S . -B build-asan \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_SANITIZER_ADDRESS=ON \
    -DENABLE_SANITIZER_UNDEFINED_BEHAVIOR=ON
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

The leaks of the MPI runtime itself are suppressed via
`cmake/sanitizers/lsan.supp`.

Because the static analyzers are enabled through `CMAKE_CXX_CLANG_TIDY` and
friends, they also run over the fetched dependencies (GoogleTest, {fmt});
diagnostics from `_deps/` paths can be ignored.

## Coverage report locally

For local coverage with GCC/Clang:

```sh
cmake -S . -B build-coverage \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_COVERAGE=ON
cmake --build build-coverage --target check-coverage
```

The HTML report is generated via `genhtml` and ends up at
`build-coverage/coverage/index.html` (not available on Windows).
