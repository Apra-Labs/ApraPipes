cmake_minimum_required(VERSION 3.22)
include_directories(
../base
${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(face_detection_cpu)
