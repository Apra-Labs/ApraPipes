cmake_minimum_required(VERSION 3.22)
set(TARGET play_mp4_from_beginning)
SET(SAMPLE_FILES
	PlayMp4VideoFromBeginning.cpp
	pipelineMain.cpp
)
SET(SAMPLE_FILES_H
	PlayMp4VideoFromBeginning.h
)
SET(SOURCE
  ${SAMPLE_FILES}
  ${SAMPLE_FILES_H}
)
add_executable(${TARGET} ${SOURCE})
target_include_directories ( ${TARGET} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
)
target_link_libraries(
  ${TARGET}
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
 