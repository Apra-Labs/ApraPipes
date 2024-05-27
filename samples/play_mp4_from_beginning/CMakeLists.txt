cmake_minimum_required(VERSION 3.22)
set(TARGET play_mp4_from_beginning)
SET(CORE_FILES
  play_mp4_video_from_beginning.cpp
  ../../base/test/test_utils.cpp
)
SET(CORE_FILES_H
	PlayMp4VideoFromBeginning.h
	../../base/test/test_utils.h
)
SET(SOURCE
  ${CORE_FILES}
  ${CORE_FILES_H}
)
add_executable(${TARGET} ${SOURCE})
target_include_directories ( ${TARGET} PRIVATE
  ${JETSON_MULTIMEDIA_LIB_INCLUDE}
  ${FFMPEG_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
  ${BARESIP_INC_DIR}
  ${LIBRE_INC_DIR}
  ${NVCODEC_INCLUDE_DIR}
)
target_link_libraries(
  ${TARGET}
  aprapipes
  ${JPEG_LIBRARIES}
  ${LIBMP4_LIB}
  ${OPENH264_LIB}
  ${Boost_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${JETSON_LIBS}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
  ${NVJPEGLIB_L4T}
  ${CURSES_LIBRARIES}
  )