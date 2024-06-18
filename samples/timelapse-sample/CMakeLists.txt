cmake_minimum_required(VERSION 3.22)
set(TARGET generateTimeLapse)
SET(SAMPLE_FILES
	timelapse_summary.cpp
  pipelineMain.cpp
)
SET(SAMPLE_FILES_H
timelapse_summary.h
)
SET(SOURCE
  ${SAMPLE_FILES}
  ${SAMPLE_FILES_H}
)
add_executable(${TARGET} ${SOURCE})
target_include_directories ( ${TARGET} PRIVATE
${JETSON_MULTIMEDIA_LIB_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
)
target_link_libraries(
  ${TARGET}
  aprapipes
  ${LIBMP4_LIB}
  ${OPENH264_LIB}
  ${Boost_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
  )