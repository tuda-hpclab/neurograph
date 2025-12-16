# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
    "${proc_count}"
    CACHE STRING "number of jobs that run the tests")

set(check_args -j${CTEST_JOB_COUNT} --output-on-failure ${CTEST_EXTRA_ARGS})

add_custom_target(
  check
  COMMAND
    ${CMAKE_COMMAND} -E env
    $<$<BOOL:${ENABLE_SANITIZER_ADDRESS}>:ASAN_OPTIONS=fast_unwind_on_malloc=0,symbolize=1>
    $<$<BOOL:${ENABLE_SANITIZER_LEAK}>:LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/cmake/sanitizers/lsan.supp>
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
# # googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.12.1)

# google test
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(graph_tests_additional_files
    ""
    CACHE INTERNAL "")

if(WIN32)
  list(APPEND graph_tests_additional_files "GraphTest.h")

  list(APPEND graph_tests_additional_files "distributed_graph/test_distributed_graph.h")

  list(APPEND graph_tests_additional_files "all_pairs_distances/test_all_pairs_distances.h")
  list(APPEND graph_tests_additional_files "all_pairs_distances_histogram/test_all_pairs_distances_histogram.h")
  list(APPEND graph_tests_additional_files "apsp/test_apsp.h")
  list(APPEND graph_tests_additional_files "arc_counter/test_arc_counter.h")
  list(APPEND graph_tests_additional_files "arc_length/test_arc_length.h")
  list(APPEND graph_tests_additional_files "arc_length_histogram/test_arc_length_histogram.h")
  list(APPEND graph_tests_additional_files "area_connectivity/test_area_connectivity.h")
  list(APPEND graph_tests_additional_files "assortativity/test_assortativity.h")
  list(APPEND graph_tests_additional_files "centrality/test_betweenness_centrality.h")
  list(APPEND graph_tests_additional_files "clustering/test_clustering_coefficient.h")
  list(APPEND graph_tests_additional_files "degree/test_degree.h")
  list(APPEND graph_tests_additional_files "degree_counter/test_degree_counter.h")
  list(APPEND graph_tests_additional_files "degree_histogram/test_degree_histogram.h")
  list(APPEND graph_tests_additional_files "delta_stepping/test_delta_stepping.h")
  list(APPEND graph_tests_additional_files "diameter_approximation/test_diameter_approximation.h")
  list(APPEND graph_tests_additional_files "histogram/test_histogram.h")
  list(APPEND graph_tests_additional_files "network_motifs/test_network_motifs.h")
  list(APPEND graph_tests_additional_files "node_counter/test_node_counter.h")
  list(APPEND graph_tests_additional_files "weight_counter/test_weight_counter.h")
endif()

add_executable(graph_tests ${SRC_TST})
target_sources(
  graph_tests
  PRIVATE GraphTest.cpp
          all_pairs_distances/test_all_pairs_distances.cpp
          all_pairs_distances_histogram/test_all_pairs_distances_histogram.cpp
          apsp/test_apsp.cpp
          arc_counter/test_arc_counter.cpp
          arc_length/test_arc_length.cpp
          arc_length_histogram/test_arc_length_histogram.cpp
          area_connectivity/test_area_connectivity.cpp
          assortativity/test_assortativity.cpp
          centrality/test_betweenness_centrality.cpp
          clustering/test_clustering_coefficient.cpp
          degree/test_degree.cpp
          degree_counter/test_degree_counter.cpp
          degree_histogram/test_degree_histogram.cpp
          delta_stepping/test_delta_stepping.cpp
          diameter_approximation/test_diameter_approximation.cpp
          distributed_graph/test_distributed_graph.cpp
          histogram/test_histogram.cpp
          network_motifs/test_network_motifs.cpp
          node_counter/test_node_counter.cpp
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

get_target_property(graph_tests_sources graph_tests SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${graph_tests_sources})
