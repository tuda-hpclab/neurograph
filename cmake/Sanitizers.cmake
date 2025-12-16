function(enable_sanitizers project_name)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES
                                             ".*Clang")
    option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)

    if(ENABLE_COVERAGE)
      if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        target_compile_options(
          ${project_name} INTERFACE -fprofile-instr-generate=prof_%p.profraw
                                    -fcoverage-mapping)
        target_link_options(
          ${project_name} INTERFACE -fprofile-instr-generate=prof_%p.profraw
          -fcoverage-mapping)
      else()
        target_compile_options(${project_name} INTERFACE --coverage -O0 -g)
        target_link_options(${project_name} INTERFACE --coverage)
      endif()
    endif()

    set(SANITIZERS "")

    option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    if(ENABLE_SANITIZER_ADDRESS)
      list(APPEND SANITIZERS "address")
    endif()

    option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    if(ENABLE_SANITIZER_LEAK)
      list(APPEND SANITIZERS "leak")
    endif()

    option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR
           "Enable undefined behavior sanitizer" OFF)
    if(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR)
      list(APPEND SANITIZERS "undefined")
    endif()

    option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    if(ENABLE_SANITIZER_THREAD)
      if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
        message(
          WARNING
            "Thread sanitizer does not work with Address and Leak sanitizer enabled"
        )
      else()
        list(APPEND SANITIZERS "thread")
      endif()
    endif()

    option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    if(ENABLE_SANITIZER_MEMORY AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      message(
        WARNING
          "Memory sanitizer requires all the code (including libc++) to be MSan-instrumented otherwise it reports false positives"
      )
      if("address" IN_LIST SANITIZERS
         OR "thread" IN_LIST SANITIZERS
         OR "leak" IN_LIST SANITIZERS)
        message(
          WARNING
            "Memory sanitizer does not work with Address, Thread and Leak sanitizer enabled"
        )
      else()
        list(APPEND SANITIZERS "memory")
      endif()
    endif()

    list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

  endif()

  if(LIST_OF_SANITIZERS)
    if(NOT "${LIST_OF_SANITIZERS}" STREQUAL "")
      target_compile_options(${project_name}
                             INTERFACE -fsanitize=${LIST_OF_SANITIZERS})
      target_link_options(${project_name} INTERFACE
                          -fsanitize=${LIST_OF_SANITIZERS})
    endif()
  endif()

endfunction()

function(enable_coverage_targets)
  if(NOT ENABLE_COVERAGE)
    return()
  endif()

  if(WIN32 AND ENABLE_COVERAGE)
    message("Cannot use coverage with WIN32")
    return()
  endif()

  set(COVERAGE_REPORT_AVAILABLE
      OFF
      PARENT_SCOPE)

  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    find_package(Python COMPONENTS Interpreter)
    if(NOT Python_Interpreter_FOUND)
      message("Python interpreter not found (required for report-combine)")
      return()
    endif()

    find_program(LLVM_PROFDATA llvm-profdata)
    if(LLVM_PROFDATA MATCHES "-NOTFOUND")
      message(
        "llvm-profdata executable not found (required for report-combine)")
      return()
    endif()

    find_program(LLVM_COV llvm-cov)
    if(LLVM_COV MATCHES "-NOTFOUND")
      message("llvm-cov executable not found (required for coverage-collect)")
      return()
    endif()

    add_custom_target(
      report-combine
      COMMAND
        ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/merge-clang-raw-profiles.py --profile-dir
        ${CMAKE_BINARY_DIR} --llvm-profdata-tool ${LLVM_PROFDATA} --output
        ${CMAKE_BINARY_DIR}/profile.profdata
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_custom_target(
      coverage-clean
      COMMAND sh -c "find ${CMAKE_BINARY_DIR} -name 'prof_*.profraw' -delete"
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  else()
    find_program(LCOV lcov)
    if(LCOV MATCHES "-NOTFOUND")
      message("lcov executable not found (required for coverage-collect)")
      return()
    endif()

    add_custom_target(coverage-clean)

    add_custom_target(
      coverage-clean-impl
      COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --zerocounters
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_custom_target(
      coverage-initial
      COMMAND
        ${LCOV} --capture --initial --branch-coverage --ignore-errors mismatch
        --directory ${CMAKE_BINARY_DIR} --output-file
        ${CMAKE_BINARY_DIR}/coverage_baseline.info
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage_baseline.info)

    add_dependencies(coverage-clean coverage-initial coverage-clean-impl)
    add_dependencies(coverage-initial coverage-clean-impl)

    add_custom_target(
      coverage-collect-impl
      COMMAND
        ${LCOV} --directory ${CMAKE_BINARY_DIR} --branch-coverage
        --ignore-errors mismatch --capture --output-file
        ${CMAKE_BINARY_DIR}/coverage_collected.info
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage_collected.info)

    add_custom_target(
      coverage-collect
      COMMAND
        ${LCOV} --branch-coverage --exclude "test" --exclude "_deps" --exclude
        "usr" --add-tracefile ${CMAKE_BINARY_DIR}/coverage_baseline.info
        --add-tracefile ${CMAKE_BINARY_DIR}/coverage_collected.info
        --output-file ${CMAKE_BINARY_DIR}/coverage.info
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage.info)

    add_dependencies(coverage-collect coverage-collect-impl)
  endif()

  if(NOT WIN32 AND "${ENABLE_COVERAGE}")
    find_program(GENHTML genhtml)
    if(GENHTML MATCHES "-NOTFOUND")
      message("genhtml executable not found (required for coverage-report)")
      return()
    endif()

    add_custom_target(
      coverage-report
      COMMAND
        ${GENHTML} --filter branch,function,line,trivial --branch-coverage
        --demangle-cpp -o coverage --prefix ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/coverage.info
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_dependencies(coverage-report coverage-collect)
    endif()

    set(COVERAGE_REPORT_AVAILABLE
        ON
        PARENT_SCOPE)
  endif()

endfunction(enable_coverage_targets)

function(add_coverage_collection target)
  if(WIN32 OR NOT ENABLE_COVERAGE)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    add_custom_target(
      coverage-collect-${target}
      COMMAND
        ${LLVM_COV} export --format=lcov
        --instr-profile=${CMAKE_BINARY_DIR}/profile.profdata
        --ignore-filename-regex="test" ${CMAKE_BINARY_DIR}/bin/${target} >
        ${CMAKE_BINARY_DIR}/coverage.info
      DEPENDS report-combine
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage.info)

    add_dependencies(coverage-report coverage-collect-${target})
  else()
    add_dependencies(coverage-clean ${target})
  endif()

endfunction(add_coverage_collection)
