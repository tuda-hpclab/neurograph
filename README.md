# neurograph

neurograph is a library of graph algorithms designed for neuroscientific
applications. It implements various graph algorithms using MPI that are
necessary for comparing structural connectomes. The structural connectome is
the aspect of the brain that relates to the connectivity of neurons. In
contrast to the functional connectome, which is defined by the neurons and
their connections, ignoring any temporal correlation of activity, the
structural connectome is defined by the physical connections between
neurons. This enables us to represent the structural connectome as a
directed graph, where neurons serve as vertices and synapses as arcs. In
this interpretation, however, we omit details such as the detailed
morphology of an axon or a dendrite, including the exact location of a
synapse. Most use cases of analyzing the structural connectome can be solved
with the [*Brain Connectivity Toolbox*](https://sites.google.com/site/bctnet/),
but this is not the case when the graphs become too large. While most other
distributed graph-processing libraries focus on algorithms required for,
e.g., planning, we follow the work of Rubinov and Sporns:
[*Complex network measures of brain connectivity: Uses and interpretations*](https://www.sciencedirect.com/science/article/pii/S105381190901074X?via%3Dihub).

neurograph itself is an MPI-parallel C++20 application built around these
algorithms, computing graph metrics on large, directed, weighted graphs
distributed across processes. Each MPI rank loads its own slice of the graph
(node positions, areas, signal types, and incoming/outgoing arcs); remote
parts are accessed via one-sided MPI communication (RMA windows), so no
single rank ever needs a copy of the full graph. Metrics are selected on the
command line. Several metrics offer interchangeable algorithm variants (e.g.
different vertex queues for Dijkstra, five epoch strategies for
Delta-Stepping, Brandes or path enumeration for betweenness centrality, four
community-detection algorithms for modularity). The root rank writes one text
file per metric to an output directory.

## Use Case

For analyses where a graph is too large or too distributed to process with
tools like NetworkX or bctpy on a single machine, while keeping results
comparable to those reference tools.

- **Distributed representation:** `DistributedGraph` holds each rank's nodes
  and arcs in MPI RMA windows. Remote adjacency lists are fetched on demand via
  passive RMA-Get and cached.
- **Broad metric coverage:** from simple counts and degree/geometry statistics
  to all-pairs shortest paths, betweenness centrality, clustering, transitivity,
  network motifs, assortativity, rich-club coefficients, area connectivity,
  modularity, strongly connected components, and maximum flow.
- **Interchangeable variants:** core algorithms are organized as dedicated
  types (`DijkstraVariants`, `DeltaSteppingVariants`,
  `BetweennessCentralityVariants`, `ModularityVariants`, `MaxFlowVariants`,
  `SccVariants`)
  selected via thin dispatchers and CLI flags — convenient for benchmarking
  variants against each other.
- **Deterministic results:** metrics produce the same values regardless of
  rank count (the only exception is the randomized diameter approximation);
  the test suite checks matching expectations at one and at four ranks.
- **Load-time graph transforms:** drop self-loops, treat arcs as undirected
  (summing weights), or set all weights to one — each via a flag, without
  touching the input files.

See [docs/functionality.md](docs/functionality.md) for the full input format
and metric reference.

## Requirements

- **A C++20 compiler.** Uses Concepts, Ranges, `std::span`, and designated
  initializers; regularly built with MSVC and GCC.
- **CMake 3.22+** (on Windows, e.g. with the Ninja generator).
- **An MPI implementation** (e.g. MS-MPI, Open MPI, MPICH). MPI is a required
  dependency; the program is launched via `mpiexec`/`mpirun`.
- **OpenMP** for select computations (`--threads`). `/openmp:llvm` is set on
  Windows; OpenMP is used on Linux if available.
- **Git and network access on first configure:** `FetchContent` downloads
  {fmt} 12.2.0, spdlog 1.17.0, range-v3 0.12.0, and CLI11 2.6.2 automatically.

The header-only libraries `cpp-utility` and `mpi-wrapper` ship with this
repository under [`external/`](external/) and need nothing downloaded, see
[`external/README.md`](external/README.md) for their origin and revision.

## Building and Testing

```sh
git clone <repo-url> neurograph
cd neurograph

cmake -S . -B build
cmake --build build

ctest --test-dir build --output-on-failure
```

The executable ends up at `build/bin/neurograph` and the test suite at
`build/bin/graph_tests`.

See [docs/building.md](docs/building.md) for development-only requirements,
CMake options, and the bundled dependencies.

## Documentation

- [docs/functionality.md](docs/functionality.md) — full feature reference
  (input file format and the metric reference: counts, degree, geometry,
  paths, local structure, community, connectivity, flow)
- [docs/building.md](docs/building.md) — development-only requirements,
  CMake options, and the bundled dependencies
- [docs/examples.md](docs/examples.md) — annotated example calls
- [docs/testing.md](docs/testing.md) — test suite, sanitizer and
  static-analysis builds, and coverage
- [docs/formatting.md](docs/formatting.md) — copyright header and include
  order conventions

## License

This project is released under a BSD-style license; see `LICENSE.dat`.
Developed at Technical University of Darmstadt.
