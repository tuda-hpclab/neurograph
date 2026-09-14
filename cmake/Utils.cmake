# Coverage report
#
# GCC only: a GCC debug tree carries the gcov data a coverage report is made of anyway,
# while clangs --coverage needs its profile runtime (libclang_rt.profile), which not every
# installation ships. Coverage with clang stays available on demand, via -DENABLE_COVERAGE=ON.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("BuildType is debug. Will collect coverage information")
    include(cmake/CodeCoverage.cmake)
    # The flags go to the targets of this project instead of CMAKE_CXX_FLAGS: the report
    # covers source/ only, and the global variable also instruments every dependency that
    # is fetched below and every compiler probe that runs at configure time.
    separate_arguments(coverage_compiler_flags NATIVE_COMMAND "${COVERAGE_COMPILER_FLAGS}")
    target_compile_options(project_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${coverage_compiler_flags}>)
    target_link_options(project_options INTERFACE ${coverage_compiler_flags})
else()
    message("Not a GCC debug build (${CMAKE_BUILD_TYPE}, ${CMAKE_CXX_COMPILER_ID}). Will not collect coverage")
endif ()