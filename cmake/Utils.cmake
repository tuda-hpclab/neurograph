# Coverage report
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("BuildType is debug. Will collect coverage information")
    include(cmake/CodeCoverage.cmake)
    append_coverage_compiler_flags()
else()
    message("Not debug. Will not collect coverage $CMAKE_BUILD_TYPE")
endif ()