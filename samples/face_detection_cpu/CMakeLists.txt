cmake_minimum_required(VERSION 3.22)
set(TARGET face_detection_cpu)

set(SAMPLE_FILES
  face_detection_cpu.cpp
  pipelineMain.cpp
)
set(SAMPLE_FILES_H
  face_detection_cpu.h
)
set(SOURCE
  ${SAMPLE_FILES}
  ${SAMPLE_FILES_H}
)

add_executable(${TARGET} ${SOURCE})

target_include_directories(${TARGET} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ../../base
)

target_link_libraries(${TARGET}
  aprapipes
  ${JPEG_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
