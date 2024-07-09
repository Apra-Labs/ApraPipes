cmake_minimum_required(VERSION 3.22)
set(TARGET face_detection_cpu)
SET(SAMPLE_FILES
  face_detection_cpu.cpp
  pipelineMain.cpp
)
SET(SAMPLE_FILES_H
	face_detection_cpu.h
)
SET(SOURCE
  ${SAMPLE_FILES}
  ${SAMPLE_FILES_H}
)

add_executable(${TARGET} ${SOURCE})

target_include_directories ( ${TARGET} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
  ${TARGET}
  aprapipes
  ${JPEG_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  )