# Copyright (c) 2012 - 2017, Lars Bilke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# CHANGES:
#
# 2012-01-31, Lars Bilke
# - Enable Code Coverage
#
# 2013-09-17, Joakim Söderberg
# - Added support for Clang.
# - Some additional usage instructions.
#
# 2016-02-03, Lars Bilke
# - Refactored functions to use named parameters
#
# 2017-06-02, Lars Bilke
# - Merged with modified version from github.com/ufz/ogs
#
# 2019-05-06, Anatolii Kurotych
# - Remove unnecessary --coverage flag
#
# 2019-12-13, FeRD (Frank Dana)
# - Deprecate COVERAGE_LCOVR_EXCLUDES and COVERAGE_GCOVR_EXCLUDES lists in favor
#   of tool-agnostic COVERAGE_EXCLUDES variable
#
# 2020-01-19, Mike Purvis
# - Add genhtml options to customise HTML output
#
# 2020-03-10, Martin Grap
# - Add option to create XML coverage reports
#
# USAGE:
#
# 1. Copy this file into your cmake modules path.
#
# 2. Add the following line to your CMakeLists.txt (best before any add_executable()
#    or add_library() calls):
#      include(CodeCoverage)
#
# 3. Append necessary compiler flags for code coverage:
#      append_coverage_compiler_flags()
#
# 3.a (OPTIONAL) Set appropriate optimization flags, e.g. -O0, -O1 or -Og
#
# 4. If you need to exclude additional directories from the report, specify them
#    using full paths in the COVERAGE_EXCLUDES variable before calling
#    setup_target_for_coverage_*().
#    Example:
#      set(COVERAGE_EXCLUDES
#          '${PROJECT_SOURCE_DIR}/tests/*'
#          '/path/to/my/directory/*')
#    Or, use the EXCLUDE argument to setup_target_for_coverage_*().
#    Example:
#      setup_target_for_coverage_lcov(
#          NAME coverage
#          EXECUTABLE testrunner
#          EXCLUDE "${PROJECT_SOURCE_DIR}/tests/*")
#
# 5. Use the functions described below to create a custom make target which
#    runs your test executable and produces a code coverage report.
#
# 6. Build a Debug build:
#      cmake -DCMAKE_BUILD_TYPE=Debug ..
#      make
#      make my_coverage_target
#

include(CMakeParseArguments)

option(CODE_COVERAGE "Enable coverage reporting" OFF)

if(CODE_COVERAGE AND NOT CODE_COVERAGE_ADDED)
    set(CODE_COVERAGE_ADDED 1)

    # Check prereqs
    find_program(GCOV_PATH gcov)
    find_program(LCOV_PATH NAMES lcov lcov.bat lcov.exe lcov.perl)
    find_program(GENHTML_PATH NAMES genhtml genhtml.perl genhtml.bat)
    find_program(GCOVR_PATH gcovr PATHS ${CMAKE_SOURCE_DIR}/scripts/test)

    if(NOT GCOV_PATH)
        message(FATAL_ERROR "gcov not found! Aborting...")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        if(CMAKE_C_COMPILER_VERSION VERSION_LESS 3)
            message(FATAL_ERROR "Clang version must be 3.0.0 or greater! Aborting...")
        endif()
    elseif(NOT CMAKE_COMPILER_IS_GNUCXX)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
            message(FATAL_ERROR "Compiler is not GNU gcc or Clang! Aborting...")
        endif()
    endif()

    set(COVERAGE_COMPILER_FLAGS "-g -fprofile-arcs -ftest-coverage"
        CACHE INTERNAL "")

    set(CMAKE_C_FLAGS_COVERAGE
        ${COVERAGE_COMPILER_FLAGS}
        CACHE STRING "Flags used by the C compiler during coverage builds."
        FORCE)
    set(CMAKE_CXX_FLAGS_COVERAGE
        ${COVERAGE_COMPILER_FLAGS}
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
        FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        ""
        CACHE STRING "Flags used for linking binaries during coverage builds."
        FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        ""
        CACHE STRING "Flags used by the shared libraries linker during coverage builds."
        FORCE)
    mark_as_advanced(
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

    if(CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_COVERAGE} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE} -fprofile-instr-generate -fcoverage-mapping")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Code coverage results with an optimised (non-Debug) build may be misleading")
    endif()

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        link_libraries(gcov)
    endif()

    # Defines a target for running and collection code coverage information
    # Builds dependencies, runs the given executable and outputs reports.
    # NOTE! The executable should always have a ZERO as exit code otherwise
    # the coverage generation will not complete.
    #
    # setup_target_for_coverage_lcov(
    #     NAME testrunner_coverage                    # New target name
    #     EXECUTABLE testrunner -j ${PROCESSOR_COUNT} # Executable in PROJECT_BINARY_DIR
    #     DEPENDENCIES testrunner                     # Dependencies to build first
    #     EXCLUDE "tests/*"                           # Patterns to exclude (can be relative to BASE_DIRECTORY)
    #     BASE_DIRECTORY "../"                         # Base directory for report
    #     NO_DEMANGLE                                  # Don't demangle C++ symbols
    #     LCOV_ARGS "--no-external"                    # Extra arguments for lcov
    #     GENHTML_ARGS "--legend"                      # Extra arguments for genhtml
    # )
    function(setup_target_for_coverage_lcov)
        set(options NO_DEMANGLE)
        set(oneValueArgs BASE_DIRECTORY NAME)
        set(multiValueArgs EXCLUDE EXECUTABLE EXECUTABLE_ARGS DEPENDENCIES LCOV_ARGS GENHTML_ARGS)
        cmake_parse_arguments(Coverage "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        if(NOT LCOV_PATH)
            message(FATAL_ERROR "lcov not found! Aborting...")
        endif()

        if(NOT GENHTML_PATH)
            message(FATAL_ERROR "genhtml not found! Aborting...")
        endif()

        # Set base directory (as absolute path), or default to PROJECT_SOURCE_DIR
        if(${Coverage_BASE_DIRECTORY})
            get_filename_component(BASEDIR ${Coverage_BASE_DIRECTORY} ABSOLUTE)
        else()
            set(BASEDIR ${PROJECT_SOURCE_DIR})
        endif()

        # Collect excludes (CMake 3.4+: Also compute absolute paths)
        set(LCOV_EXCLUDES "")
        foreach(EXCLUDE ${Coverage_EXCLUDE} ${COVERAGE_EXCLUDES} ${COVERAGE_LCOV_EXCLUDES})
            if(CMAKE_VERSION VERSION_GREATER "3.4")
                get_filename_component(EXCLUDE ${EXCLUDE} ABSOLUTE BASE_DIR ${BASEDIR})
            endif()
            list(APPEND LCOV_EXCLUDES "${EXCLUDE}")
        endforeach()
        list(REMOVE_DUPLICATES LCOV_EXCLUDES)

        # Conditional arguments
        if(CPPFILT_PATH AND NOT ${Coverage_NO_DEMANGLE})
            set(GENHTML_EXTRA_ARGS "--demangle-cpp")
        endif()

        # Setup target
        add_custom_target(${Coverage_NAME}
            # Cleanup lcov
            COMMAND ${LCOV_PATH} ${Coverage_LCOV_ARGS} --gcov-tool ${GCOV_PATH} --directory . --zerocounters
            # Create baseline to make sure untouched files show up in the report
            COMMAND ${LCOV_PATH} ${Coverage_LCOV_ARGS} --gcov-tool ${GCOV_PATH} --directory . --capture --initial --output-file ${Coverage_NAME}.base
            # Run tests
            COMMAND ${Coverage_EXECUTABLE} ${Coverage_EXECUTABLE_ARGS}
            # Capturing lcov counters and generating report
            COMMAND ${LCOV_PATH} ${Coverage_LCOV_ARGS} --gcov-tool ${GCOV_PATH} --directory . --capture --output-file ${Coverage_NAME}.capture
            # add baseline counters
            COMMAND ${LCOV_PATH} ${Coverage_LCOV_ARGS} --gcov-tool ${GCOV_PATH} --add-tracefile ${Coverage_NAME}.base --add-tracefile ${Coverage_NAME}.capture --output-file ${Coverage_NAME}.total
            # filter collected data to final coverage report
            COMMAND ${LCOV_PATH} ${Coverage_LCOV_ARGS} --gcov-tool ${GCOV_PATH} --remove ${Coverage_NAME}.total ${LCOV_EXCLUDES} --output-file ${Coverage_NAME}.info
            # Generate HTML output
            COMMAND ${GENHTML_PATH} ${GENHTML_EXTRA_ARGS} ${Coverage_GENHTML_ARGS} -o ${Coverage_NAME} ${Coverage_NAME}.info
            # Cleanup
            COMMAND ${CMAKE_COMMAND} -E remove ${Coverage_NAME}.base ${Coverage_NAME}.capture ${Coverage_NAME}.total
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            DEPENDS ${Coverage_DEPENDENCIES}
            COMMENT "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
        )

        # Show where to find the lcov info report
        add_custom_command(TARGET ${Coverage_NAME} POST_BUILD
            COMMAND ;
            COMMENT "Lcov code coverage info report saved in ${Coverage_NAME}.info."
        )

        # Show info where to find the report
        add_custom_command(TARGET ${Coverage_NAME} POST_BUILD
            COMMAND ;
            COMMENT "Open ${PROJECT_BINARY_DIR}/${Coverage_NAME}/index.html in your browser to view the coverage report."
        )

    endfunction()

    function(append_coverage_compiler_flags)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_COMPILER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}" PARENT_SCOPE)
        message(STATUS "Appending code coverage compiler flags: ${COVERAGE_COMPILER_FLAGS}")
    endfunction()

    function(append_coverage_compiler_flags_to_target name)
        separate_arguments(_flag_list NATIVE_COMMAND "${COVERAGE_COMPILER_FLAGS}")
        target_compile_options(${name} PRIVATE ${_flag_list})
        target_link_libraries(${name} PRIVATE gcov)
    endfunction()

endif()
