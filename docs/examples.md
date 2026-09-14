# Examples

All examples run on the bundled `rank_8_nodes_5000_arcs_6` input graph; see
[functionality.md](functionality.md) for the full metric and option reference.

## Basic counts and degree

Basic metrics on the bundled 5,000-node-per-rank example network with eight
ranks:

```sh
mpiexec -n 8 build/bin/neurograph \
    --input input/rank_8_nodes_5000_arcs_6 \
    --func node-count,arc-count,weight-count,degree-extremes
```

## All-pairs shortest paths with Delta-Stepping

All-pairs shortest paths with Delta-Stepping instead of Dijkstra:

```sh
mpiexec -n 8 build/bin/neurograph \
    --input input/rank_8_nodes_5000_arcs_6 \
    --func avg-apsp \
    --apsp-algorithm delta-stepping --apsp-delta 10 --apsp-delta-epoch push-check
```

## Betweenness centrality and modularity

Betweenness centrality and modularity, the latter with Leiden:

```sh
mpiexec -n 8 build/bin/neurograph \
    --input input/rank_8_nodes_5000_arcs_6 \
    --func avg-centrality,modularity \
    --centrality-algorithm brandes --modularity-algorithm leiden
```

## Undirected, unweighted analysis

Analyzing the graph as an undirected, unweighted network:

```sh
mpiexec -n 8 build/bin/neurograph \
    --input input/rank_8_nodes_5000_arcs_6 \
    --func avg-cluster,assortativity \
    --undirected --all-weights-one --remove-autapses
```

## Maximum flow

Maximum flow from node 0 on rank 0 to node 3 on rank 1, computed with the
distributed push-relabel variant:

```sh
mpiexec -n 8 build/bin/neurograph \
    --input input/rank_8_nodes_5000_arcs_6 \
    --func max-flow \
    --max-flow-sources 0:0 --max-flow-sinks 1:3 \
    --max-flow-algorithm distributed-push-relabel
```
