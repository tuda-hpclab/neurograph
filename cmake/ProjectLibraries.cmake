# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# everything fetched here is a dependency, its warnings are none of our business: the
# -Werror that CMAKE_COMPILE_WARNING_AS_ERROR puts on every target is meant for the code
# of this project, and a new warning of a newer compiler must not break the build
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

add_library(project_libraries INTERFACE)

include(FetchContent)

find_package(Threads REQUIRED)
target_link_libraries(project_libraries INTERFACE Threads::Threads)

if(WIN32)
  target_compile_options(project_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
else()
  find_package(OpenMP)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(project_options INTERFACE OpenMP::OpenMP_CXX)
endif()

if(UNIX)
  target_link_libraries(project_options INTERFACE stdc++fs)
endif()

find_package(MPI REQUIRED)
target_compile_definitions(project_options
                           INTERFACE -DMPI_FOUND=$<BOOL:${MPI_CXX_FOUND}>)

# fix CI build issue
get_target_property(mpi_cxx_compile_options MPI::MPI_CXX
                    INTERFACE_COMPILE_OPTIONS)

if("${mpi_cxx_compile_options}" MATCHES "-flto=auto")
  message(
    WARNING
      "MPI_CXX was compiled with -flto=auto and -ffat-lto-objects, removing lto flags to prevent CI build failure"
  )
  string(REPLACE "-flto=auto" "" mpi_cxx_compile_options
                 ${mpi_cxx_compile_options})
  string(REPLACE "-ffat-lto-objects" "" mpi_cxx_compile_options
                 ${mpi_cxx_compile_options})
endif()

set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS
                                              "${mpi_cxx_compile_options}")

target_link_libraries(project_libraries INTERFACE MPI::MPI_CXX)

# declaration
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
cmake_policy(SET CMP0077 NEW)

# fmt
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG 12.2.0)

# spdlog
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.17.0)

# range-v3
FetchContent_Declare(
        range-v3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3
        GIT_TAG 0.12.0)

# make available

# fmt
FetchContent_MakeAvailable(fmt)
get_target_property(fmt_includes fmt INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(fmt PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                     "${fmt_includes}")
target_link_libraries(project_libraries INTERFACE fmt)

# spdlog
set(SPDLOG_FMT_EXTERNAL ON)
FetchContent_MakeAvailable(spdlog)
get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                        "${spdlog_includes}")
set(SPDLOG_FMT_EXTERNAL
    ON
    CACHE BOOL "" FORCE)
target_link_libraries(project_libraries INTERFACE spdlog)

# range-v3
FetchContent_MakeAvailable(range-v3)
get_target_property(range-v3_includes range-v3 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(range-v3 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${range-v3_includes}")
target_link_libraries(project_libraries INTERFACE range-v3)


# cpp-utility and mpi-wrapper are header-only and ship with this repository, see
# external/README.md. SYSTEM, like every other dependency: their warnings are none of
# our business, and they are deliberately built with a different warning set upstream
# (-Wuseless-cast, for one, is off there because it does not play nice with std::size_t).
target_include_directories(project_libraries SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/external)


# set compile commands and warnings as errors back to on
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_COMPILE_WARNING_AS_ERROR)
