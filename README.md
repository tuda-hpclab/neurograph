# neurograph
neurograph is a library of graph algorithms designed for neuroscientific applications. 
It implements various graph algorithms using MPI that are necessary for comparing structural connectomes. 
The structural connectome is the aspect of the brain that relates to the connectivity of neurons. 
In contrast to the functional connectome, which is defined by the neurons and their connections, ignoring any temporal correlation of activity, the structural connectome is defined by the physical connections between neurons.
This enables us to represent the structural connectome as a directed graph, where neurons serve as vertices and synapses as arcs.
In this interpretation, however, we omit details such as the detailed morphology of an axon or a dendrite, including the exact location of a synapse.
Most use cases of analyzing the structural connectome can be solved with the [*Brain Connectivity Toolbox*](https://sites.google.com/site/bctnet/), but this is not the case when the graphs become too large.
While most other distributed graph-processing libraries focus on algorithms required for, e.g., planning, we follow the work of Rubinov and Sporns: [*Complex network measures of brain connectivity: Uses and interpretations*](https://www.sciencedirect.com/science/article/pii/S105381190901074X?via%3Dihub).

## Installation
Using the project requires CMake and a C++ compiler capable of C++20, as well as an MPI library.

## Dependencies
- Parsing the command line arguments is done with CLI11
- Logging is done with spdlog
- Tests are written with GoogleTest

## Tested module combination
- gcc/14.2.0
- openmpi/4.1.7
- cmake/3.30.5
