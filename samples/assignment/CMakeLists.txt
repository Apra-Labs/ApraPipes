cmake_minimum_required(VERSION 3.22)
project(Assignment)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

 
set(TARGET Assignment)

set(SAMPLE_FILES
    circle.cpp
    rectangle.cpp
    triangle.cpp
    pipelineMain.cpp
)

set(SAMPLE_HEADERS
    Shape.h
    circle.h
    rectangle.h
    triangle.h
)

set(SOURCE
    ${SAMPLE_FILES}
    ${SAMPLE_HEADERS}
)

 
# --- Main application ---
add_executable(Assignment
    pipelineMain.cpp
    circle.cpp
    triangle.cpp
    rectangle.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}        
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${LIBMP4_INC_DIR}
)

target_link_libraries(${TARGET}
    aprapipes
    ${JPEG_LIBRARIES}
    ${LIBMP4_LIB}
    ${OPENH264_LIB}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${NVCUDAToolkit_LIBS}
    ${NVJPEGLIB_L4T}
    ${NVCODEC_LIB}
    ${FFMPEG_LIBRARIES}
)



# --- Unit test (separate exe) ---
add_executable(test_assignment
    test.cpp
    circle.cpp
    triangle.cpp
    rectangle.cpp
)


target_include_directories(test_assignment PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}       
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(test_assignment
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_test(NAME AssignmentTest COMMAND test_assignment)
