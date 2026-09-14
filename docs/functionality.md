# Functionality

## Input Format

An input directory contains the subfolders `positions/` and `network/`. Files
are split per rank; the number of MPI ranks at startup must match the file
split (`rank_0`, ..., `rank_<N-1>`, zero-padded above ten ranks).

- `positions/rank_<r>_positions.txt` — one node per line: local ID
  (1-based), position (`x y z`), area name, and signal type. Lines starting
  with `#` are comments:

  ```text
  # ID	Position (x y z)	Area	type
  1	19.37056 54.46747 63.90031	lh.bankssts	ex
  2	17.84999 54.50196 64.24411	lh.bankssts	ex
  ```

- `network/rank_<r>_in_network.txt` and `network/rank_<r>_out_network.txt` —
  arcs ending on / starting from the rank, respectively:

  ```text
  # <target rank> <target neuron id>	<source rank> <source neuron id>	<weight>
  0 1	0 2	1
  ```

  Both directions must be mutually consistent; the loader validates rank and
  ID ranges against the node distribution. `--prefix` inserts an extra name
  segment between rank and suffix (e.g. for networks from different
  simulation steps).

Examples live under [`input/`](../input/), one per rank count:
`rank_1_nodes_5000_arcs_6`, `rank_2_...`, `rank_4_...`, and
`rank_8_nodes_5000_arcs_6`, each with 5000 nodes and 6 arcs per node and per
rank.

## Metrics

Metrics are selected via `--func` as a comma-separated list.

### Counts (`metrics/counting`)

- `node-count` — total number of nodes.
- `arc-count` — total number of incoming and outgoing arcs.
- `weight-count` — total weight of incoming and outgoing arcs.

### Degree (`metrics/degree`)

- `degree-extremes` — minimum and maximum in-/out-degree.
- `degree-histogram-width` — degree histograms with a fixed bin width
  (`--bin-width-degree-histogram`).
- `degree-histogram-count` — degree histograms with bin width one over the
  full value range.

### Geometry (`metrics/geometry`)

Operates on Euclidean distances between node positions:

- `arc-length-average` — mean arc length.
- `arc-length-histogram-width`, `arc-length-histogram-count` — arc-length
  histograms with a fixed bin width or fixed bin count.
- `pair-distances-average` — mean distance across all node pairs.
- `pair-distances-histogram-width`, `pair-distances-histogram-count` —
  histograms of all pairwise distances.

### Paths (`metrics/paths`)

- `avg-apsp` — all-pairs shortest paths over arc weights. Returns mean
  shortest-path length, mean efficiency, diameter, number of disconnected
  pairs, and connected-component sizes. The SSSP core is selectable:
  - `--apsp-algorithm dijkstra` (default) with `--dijkstra-queue
    priority-queue|fibonacci-heap|bucket-queue`,
  - `--apsp-algorithm delta-stepping` with `--apsp-delta` (bucket width) and
    `--apsp-delta-epoch push-brute-force|push-check|push-short-long|`
    `push-inner-short|pull-model`.
- `avg-apsp-inv` — the same metrics over inverse weights (distance 1/w,
  Dijkstra only; `bucket-queue` is not allowed here).
- `avg-centrality` — mean betweenness centrality;
  `--centrality-algorithm brandes` (default, Brandes 2001) or
  `path-enumeration` (materializes all shortest paths explicitly).
- `dia-approx` — randomized diameter approximation via ball contraction
  around randomly chosen nodes.

### Local structure (`metrics/local_structure`)

- `avg-cluster` — mean clustering coefficient (two counting variants).
- `tri-motifs` — frequencies of the thirteen connected three-node network
  motifs, numbered after Kaiser (2011). Returns the total number of connected
  triples and the fraction of each motif. Self-loops are ignored. The algorithm
  is selectable via `--tri-motifs-algorithm
  questions|rma|rma-accumulate|rma-fair|rma-fewer-arcs|rma-fewer-messages`
  (`questions` is the default all-round choice). All variants are deterministic
  and report the same fractions; they only differ in how a rank learns whether
  the two outer nodes of a triple are connected, which is the one thing it
  cannot answer for nodes it does not own. `questions` asks their owner in
  collective question-and-answer rounds and therefore needs no remote arc
  reads at all, while the `rma-*` variants read the arcs through the RMA
  windows of the graph, which removes the collective rounds but can download
  an adjacency that another rank already holds. Among those, `rma` is the base
  variant, `rma-fair` spreads the triangles that span three ranks evenly over
  their three owners instead of giving them all to the owner of the smallest
  node, `rma-fewer-arcs` downloads the smaller of the two candidate
  adjacencies at the price of four extra remote accesses per pair,
  `rma-fewer-messages` leaves the triples whose outer nodes sit together on
  one other rank to that rank, and `rma-accumulate` replaces the final
  collective reduction with a one-sided `MPI_Accumulate`.
- `assortativity` — the four degree assortativities (in/in, in/out, out/in,
  out/out) over weighted degrees.
- `reciprocity` — the fraction of arcs whose opposite arc exists as well, plus
  the density-corrected coefficient per Garlaschelli & Loffredo (2004), which
  states whether that fraction is above (positive) or below (negative) the one
  of a random graph of the same density. Self-loops and weights are ignored.
- `rich-club` — the rich-club coefficients per Zhou & Mondragón (2004): for
  every degree threshold k, the arc density among the nodes of a degree above k
  (their club), together with the club's size and arc count. A curve that rises
  with k means the hubs are more densely connected among themselves than the
  graph is on average. The degree is the number of in arcs plus the number of
  out arcs, so a self-loop counts twice but never as an arc inside a club;
  weights are ignored. The thresholds end above the last club of two nodes,
  below which the density is undefined.
- `transitivity` — the global clustering coefficient: the closed triangles of
  all nodes divided by the triangles their neighborhoods allow, for the same two
  counting variants as `avg-cluster`. Where `avg-cluster` averages the ratio of
  every node and thereby weighs a node of degree two as much as a hub,
  `transitivity` divides the two sums and thereby weighs every node by the
  number of triangles it can be part of. On a symmetric graph, both variants are
  the classic 3 * (number of triangles) / (number of triples). Self-loops and
  weights are ignored.

### Community (`metrics/community`)

- `area-connec` — connection strength between all pairs of named areas.
- `modularity` — detects communities and computes directed modularity per
  Leicht & Newman (2008) for the found partition (identical to classic
  undirected modularity on symmetric graphs). The detection algorithm is
  selectable via `--modularity-algorithm louvain` (default, Blondel et al.
  2008), `leiden` (Traag et al. 2019), `label-propagation` (Raghavan et al.
  2007), or `infomap` (Rosvall & Bergstrom 2008). All variants are
  deterministic; weights are used as absolute values. Only Louvain and Leiden
  maximize the modularity itself — label propagation just follows the local
  weight majority and Infomap minimizes the description length of a random
  walk, so both usually report a lower value for their (differently motivated)
  partition.

### Connectivity (`metrics/connectivity`)

- `scc` — the strongly connected components, i.e. the maximal node sets in
  which every node reaches every other one. Returns their number, the size of
  the largest one, and how many components exist per size. Self-loops and
  weights are ignored, so a node without arcs is a component of its own. The
  algorithm is selectable via `--scc-algorithm
  tarjan-gathered|forward-backward|coloring|multistep` (`multistep` is the
  default all-round choice). All variants are deterministic and find the same
  components: `tarjan-gathered` gathers the arcs on the root rank and runs
  Tarjan (1972) there, `forward-backward` peels one component per round on the
  distributed graph (Fleischer et al. 2000), `coloring` peels a whole color
  layer per round (Orzan 2004), and `multistep` combines trimming, one
  forward-backward round, and the coloring rounds (Slota et al. 2014). For
  `multistep`, `--scc-tail-threshold <n>` additionally lets the root rank
  finish the last `n` nodes sequentially, which trades memory on that rank for
  fewer collective rounds; the default of `0` keeps every phase distributed.

### Flow (`metrics/flow`)

- `max-flow` — maximum flow value from a set of source nodes to a set of sink
  nodes (joined through a super source/sink if there are several); arc
  capacities are the absolute value of their weight, self-loops carry no
  flow. Requires `--max-flow-sources` and `--max-flow-sinks`, each a
  comma-separated list of `rank:node_id` pairs. The algorithm is selectable
  via `--max-flow-algorithm ford-fulkerson|edmonds-karp|dinic|`
  `orlin-king-rao-tarjan|distributed-push-relabel` (`dinic` is the default,
  strongly polynomial all-round choice). All variants are deterministic and
  compute the same value; the first four gather the arcs on the root rank
  and compute the flow there, while `distributed-push-relabel` keeps the
  graph distributed and computes the flow with all ranks.

### General options

| Option | Purpose |
| --- | --- |
| `--input <dir>` | Input directory (required) |
| `--func <list>` | Comma-separated list of metrics |
| `--threads <n>` | Number of OpenMP threads for select computations |
| `--remove-autapses` | Remove self-loops on load |
| `--undirected` | Treat all arcs as undirected (sums weights) |
| `--all-weights-one` | Set all weights to one |
| `--prefix <s>` | Extra name segment for network files |
| `--output-directory <dir>` | Subfolder under `<input>/metrics` |
| `--disable-status-report` | Suppress progress output |
| `--disable-timer-report` | Suppress timing output |

The root rank writes one text file per metric (e.g. `modularity.txt`,
`node_count.txt`) to `<input>/metrics/` or the subdirectory chosen via
`--output-directory`.
