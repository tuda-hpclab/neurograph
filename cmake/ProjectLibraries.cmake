# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

add_library(project_libraries INTERFACE)

include(FetchContent)

find_package(Threads REQUIRED)
target_link_libraries(project_libraries INTERFACE Threads::Threads)

if(WIN32)
  add_compile_options("/openmp:llvm")
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
        GIT_TAG 11.0.2)

# spdlog
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.15.0)

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

# set compile commands back to on
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
