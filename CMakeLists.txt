cmake_minimum_required(VERSION 3.22)
project(ScalableGraphAlgorithm)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

# project options
add_library(project_options INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# Add linker configuration
include(cmake/Linker.cmake)
configure_linker(project_options)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)
enable_coverage_targets()

# instrumentation support
include(cmake/Instrumentation.cmake)

# make libraries available
include(cmake/ProjectLibraries.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# utilities
include(cmake/Utils.cmake)

add_subdirectory(source)

option(SCA_ENABLE_TESTING "Enables the testing of the scalable graph algorithms" ON)

if(SCA_ENABLE_TESTING MATCHES ON)
	message("Using tests for the scalable graph algorithms")
	enable_testing()
	add_subdirectory(tests)
else()
	message("Not using tests for the scalable graph algorithms")
endif()
