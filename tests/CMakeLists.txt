# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
    "${proc_count}"
    CACHE STRING "number of jobs that run the tests")

# --output-on-failure puts the failing test's output into the log, --output-junit additionally
# reports which test failed in a machine-readable way, which is what the CI turns into its test
# report; without it a failing run only shows up as a non-zero exit code of ctest
set(check_args
    -j${CTEST_JOB_COUNT}
    --output-on-failure
    --output-junit "${CMAKE_BINARY_DIR}/test_results.xml"
    ${CTEST_EXTRA_ARGS})

add_custom_target(
  check
  COMMAND
    ${CMAKE_COMMAND} -E env
    $<$<BOOL:${ENABLE_SANITIZER_ADDRESS}>:ASAN_OPTIONS=fast_unwind_on_malloc=0:symbolize=1:detect_leaks=0>
    $<$<BOOL:${ENABLE_SANITIZER_LEAK}>:LSAN_OPTIONS=suppressions=${PROJECT_SOURCE_DIR}/cmake/sanitizers/lsan.supp>
    ${CMAKE_CTEST_COMMAND} ${check_args}
  USES_TERMINAL)

if(${COVERAGE_REPORT_AVAILABLE})
  add_custom_target(
    check-coverage
    DEPENDS check
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
            coverage-report)
  add_dependencies(check coverage-clean)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# googletest is a dependency, its warnings are none of our business: clang 21 warns about
# a char8_t conversion in gtest-printers.h, which -Werror would turn into a build failure
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

# # googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.17.0)

# google test
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_COMPILE_WARNING_AS_ERROR)

# clang 21 warns about a char8_t to char32_t conversion inside gtest-printers.h. The negative
# option only exists where the warning does, so it is tested through its positive form.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wcharacter-conversion NEUROGRAPH_HAVE_WCHARACTER_CONVERSION)
if(NEUROGRAPH_HAVE_WCHARACTER_CONVERSION)
  foreach(googletest_target IN ITEMS gtest gtest_main gmock gmock_main)
    if(TARGET ${googletest_target})
      target_compile_options(${googletest_target} PRIVATE -Wno-character-conversion)
    endif()
  endforeach()
endif()

set(graph_tests_additional_files
    ""
    CACHE INTERNAL "")

if(WIN32)
  list(APPEND graph_tests_additional_files "GraphTest.h")

  list(APPEND graph_tests_additional_files "distributed_graph/test_distributed_graph.h")

  list(APPEND graph_tests_additional_files "all_pairs_distances/test_all_pairs_distances.h")
  list(APPEND graph_tests_additional_files "all_pairs_distances_histogram/test_all_pairs_distances_histogram.h")
  list(APPEND graph_tests_additional_files "apsp/test_apsp.h")
  list(APPEND graph_tests_additional_files "apsp/test_apsp_delta_stepping.h")
  list(APPEND graph_tests_additional_files "arc/test_arc.h")
  list(APPEND graph_tests_additional_files "arc_counter/test_arc_counter.h")
  list(APPEND graph_tests_additional_files "arc_length/test_arc_length.h")
  list(APPEND graph_tests_additional_files "arc_length_histogram/test_arc_length_histogram.h")
  list(APPEND graph_tests_additional_files "arc_transformer/test_arc_transformer.h")
  list(APPEND graph_tests_additional_files "area_connectivity/test_area_connectivity.h")
  list(APPEND graph_tests_additional_files "assortativity/test_assortativity.h")
  list(APPEND graph_tests_additional_files "centrality/test_betweenness_centrality.h")
  list(APPEND graph_tests_additional_files "centrality/test_brandes.h")
  list(APPEND graph_tests_additional_files "centrality/test_path_enumeration.h")
  list(APPEND graph_tests_additional_files "clustering/test_clustering_coefficient.h")
  list(APPEND graph_tests_additional_files "degree/test_degree.h")
  list(APPEND graph_tests_additional_files "degree_counter/test_degree_counter.h")
  list(APPEND graph_tests_additional_files "degree_histogram/test_degree_histogram.h")
  list(APPEND graph_tests_additional_files "delta_stepping/seven_rank_expectations.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_pull_model.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_push_brute_force.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_push_check.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_push_inner_short.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_push_short_long.h")
  list(APPEND graph_tests_additional_files "diameter_approximation/test_diameter_approximation.h")
  list(APPEND graph_tests_additional_files "dijkstra/test_bucket_queue.h")
  list(APPEND graph_tests_additional_files "dijkstra/test_fibonacci_heap.h")
  list(APPEND graph_tests_additional_files "dijkstra/test_priority_queue.h")
  list(APPEND graph_tests_additional_files "file_loader/test_file_loader.h")
  list(APPEND graph_tests_additional_files "max_flow/test_dinic.h")
  list(APPEND graph_tests_additional_files "max_flow/test_distributed_push_relabel.h")
  list(APPEND graph_tests_additional_files "max_flow/test_edmonds_karp.h")
  list(APPEND graph_tests_additional_files "max_flow/test_ford_fulkerson.h")
  list(APPEND graph_tests_additional_files "max_flow/test_orlin_king_rao_tarjan.h")
  list(APPEND graph_tests_additional_files "modularity/test_infomap.h")
  list(APPEND graph_tests_additional_files "modularity/test_label_propagation.h")
  list(APPEND graph_tests_additional_files "modularity/test_leiden.h")
  list(APPEND graph_tests_additional_files "modularity/test_louvain.h")
  list(APPEND graph_tests_additional_files "network_motifs/network_motif_expectations.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_motif_structure.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_network_motifs.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_rma.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_rma_accumulate.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_rma_fair.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_rma_fewer_arcs.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_rma_fewer_messages.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_sequential.h")
  list(APPEND graph_tests_additional_files "node_counter/test_node_counter.h")
  list(APPEND graph_tests_additional_files "reciprocity/test_reciprocity.h")
  list(APPEND graph_tests_additional_files "rich_club/test_rich_club.h")
  list(APPEND graph_tests_additional_files "scc/test_coloring.h")
  list(APPEND graph_tests_additional_files "scc/test_forward_backward.h")
  list(APPEND graph_tests_additional_files "scc/test_multistep.h")
  list(APPEND graph_tests_additional_files "scc/test_tarjan_gathered.h")
  list(APPEND graph_tests_additional_files "transitivity/test_transitivity.h")
  list(APPEND graph_tests_additional_files "types/test_types.h")
  list(APPEND graph_tests_additional_files "weight_counter/test_weight_counter.h")
endif()

add_executable(graph_tests)
target_sources(
  graph_tests
  PRIVATE GraphTest.cpp
          GraphTestFourRanks.cpp
          GraphTestMultiComponent.cpp
          GraphTestOneRank.cpp
          GraphTestSevenRanks.cpp
          all_pairs_distances/test_all_pairs_distances.cpp
          all_pairs_distances_histogram/test_all_pairs_distances_histogram.cpp
          apsp/test_apsp.cpp
          apsp/test_apsp_delta_stepping.cpp
          arc/test_arc.cpp
          arc_counter/test_arc_counter.cpp
          arc_length/test_arc_length.cpp
          arc_length_histogram/test_arc_length_histogram.cpp
          arc_transformer/test_arc_transformer.cpp
          area_connectivity/test_area_connectivity.cpp
          assortativity/test_assortativity.cpp
          centrality/test_betweenness_centrality.cpp
          centrality/test_brandes.cpp
          centrality/test_path_enumeration.cpp
          clustering/test_clustering_coefficient.cpp
          degree/test_degree.cpp
          degree_counter/test_degree_counter.cpp
          degree_histogram/test_degree_histogram.cpp
          delta_stepping/test_pull_model.cpp
          delta_stepping/test_push_brute_force.cpp
          delta_stepping/test_push_check.cpp
          delta_stepping/test_push_inner_short.cpp
          delta_stepping/test_push_short_long.cpp
          diameter_approximation/test_diameter_approximation.cpp
          dijkstra/test_bucket_queue.cpp
          dijkstra/test_fibonacci_heap.cpp
          dijkstra/test_priority_queue.cpp
          distributed_graph/test_distributed_graph.cpp
          distributed_graph/test_seven_rank_graphs.cpp
          file_loader/test_file_loader.cpp
          max_flow/test_dinic.cpp
          max_flow/test_distributed_push_relabel.cpp
          max_flow/test_edmonds_karp.cpp
          max_flow/test_ford_fulkerson.cpp
          max_flow/test_orlin_king_rao_tarjan.cpp
          modularity/test_infomap.cpp
          modularity/test_label_propagation.cpp
          modularity/test_leiden.cpp
          modularity/test_louvain.cpp
          network_motifs/test_motif_structure.cpp
          network_motifs/test_network_motifs.cpp
          network_motifs/test_rma.cpp
          network_motifs/test_rma_accumulate.cpp
          network_motifs/test_rma_fair.cpp
          network_motifs/test_rma_fewer_arcs.cpp
          network_motifs/test_rma_fewer_messages.cpp
          network_motifs/test_sequential.cpp
          node_counter/test_node_counter.cpp
          reciprocity/test_reciprocity.cpp
          rich_club/test_rich_club.cpp
          scc/test_coloring.cpp
          scc/test_forward_backward.cpp
          scc/test_multistep.cpp
          scc/test_tarjan_gathered.cpp
          transitivity/test_transitivity.cpp
          types/test_types.cpp
          weight_counter/test_weight_counter.cpp
          # for visual studio
          ${graph_tests_additional_files})

target_include_directories(graph_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(graph_tests PRIVATE project_options project_libraries)
target_link_libraries(graph_tests PRIVATE graph_lib)
target_link_libraries(graph_tests PRIVATE gtest)

add_coverage_collection(graph_tests)

function(add_mpi_test)
  set(oneValueArgs TARGET NUM_RANKS)
  cmake_parse_arguments(ADD_MPI_TEST "" "${oneValueArgs}" "" ${ARGN})

  if(NOT MPIEXEC_EXECUTABLE)
    message(
      WARNING
        "Could not find MPI executable mpirun or mpiexec. Will no generate test MPITest_${ADD_MPI_TEST_TARGET}_${ADD_MPI_TEST_NUM_RANKS}."
    )
    return()
  endif()
  file(REAL_PATH ${MPIEXEC_EXECUTABLE} MPIEXEC_EXECUTABLE)

  list(APPEND MPITest_ARGS -n ${ADD_MPI_TEST_NUM_RANKS}
       ${CMAKE_BINARY_DIR}/bin/${ADD_MPI_TEST_TARGET})

  add_test(NAME MPITest_${ADD_MPI_TEST_TARGET}_${ADD_MPI_TEST_NUM_RANKS}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPITest_ARGS})

  set_tests_properties(MPITest_${ADD_MPI_TEST_TARGET}_${ADD_MPI_TEST_NUM_RANKS}
                       PROPERTIES PROCESSORS ${ADD_MPI_TEST_NUM_RANKS})

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_tests_properties(
      MPITest_${ADD_MPI_TEST_TARGET}_${ADD_MPI_TEST_NUM_RANKS}
      PROPERTIES RUN_SERIAL ON)
  endif()
  add_dependencies(check ${ADD_MPI_TEST_TARGET})
endfunction()

add_mpi_test(TARGET graph_tests NUM_RANKS 1)
add_mpi_test(TARGET graph_tests NUM_RANKS 4)
add_mpi_test(TARGET graph_tests NUM_RANKS 7)

get_target_property(graph_tests_sources graph_tests SOURCES)
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${graph_tests_sources})
