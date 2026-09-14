# Building

## Development-only requirements

In addition to the requirements in the main [README](../README.md):

- **GoogleTest 1.17.0**, provided via `FetchContent`.
- **`mpiexec`/`mpirun`** to run the MPI tests (CTest runs the suite with one
  and with four ranks).
- **lcov/genhtml** for coverage reports under GCC/Clang.
- **Doxygen** for the optional `doxygen-docs` target.
- Optionally **clang-tidy**, **cppcheck**, **include-what-you-use**,
  **ccache**, or **Score-P** for static analysis, faster builds, and
  instrumentation.

## Build and test

```sh
git clone <repo-url> neurograph
cd neurograph

cmake -S . -B build
cmake --build build

# Full test suite (runs graph_tests via mpiexec with 1 and 4 ranks):
ctest --test-dir build --output-on-failure

# Alternatively: build tests and run them in parallel:
cmake --build build --target check
```

The executable ends up at `build/bin/neurograph` and the test suite at
`build/bin/graph_tests`.

## CMake options

| Option | Purpose |
| --- | --- |
| `NEUROGRAPH_ENABLE_TESTING` | Configure the test suite (default: `ON`) |
| `ENABLE_COVERAGE` | Enable GCC/Clang coverage instrumentation |
| `ENABLE_SANITIZER_ADDRESS` | Enable AddressSanitizer |
| `ENABLE_SANITIZER_LEAK` | Enable LeakSanitizer |
| `ENABLE_SANITIZER_UNDEFINED_BEHAVIOR` | Enable UndefinedBehaviorSanitizer |
| `ENABLE_SANITIZER_THREAD` | Enable ThreadSanitizer |
| `ENABLE_SANITIZER_MEMORY` | Enable MemorySanitizer (Clang only) |
| `ENABLE_DOXYGEN` | Generate API docs via the `doxygen-docs` target |
| `ENABLE_CLANG_TIDY`, `ENABLE_CPPCHECK` | Enable static analysis |
| `ENABLE_CACHE` | Use `ccache`/`sccache` (default: `ON`) |
| `ENABLE_IPO` | Enable interprocedural optimization (LTO) |
| `ENABLE_SCOREP` | Enable Score-P instrumentation |

See [testing.md](testing.md) for what the sanitizer and static-analysis builds
add and how to run the tests under them.

## Bundled dependencies

`cpp-utility` and `mpi-wrapper` are header-only and ship with this repository
under [`external/`](../external/), which is added as a single `SYSTEM` include
directory. Nothing is downloaded for them; see
[`external/README.md`](../external/README.md) for their origin and revision, and
for how to update them.

{fmt}, spdlog, range-v3, CLI11, and GoogleTest are still fetched at configure
time via `FetchContent`, so the first configure needs git and network access.
