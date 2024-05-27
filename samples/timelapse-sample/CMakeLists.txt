cmake_minimum_required(VERSION 3.22)
set(TARGET timelapse-sample)
SET(CORE_FILES
	timelapse_summary.cpp
)
SET(CORE_FILES_H
	timelapse_summary.h
)
SET(SOURCE
  ${CORE_FILES}
  ${CORE_FILES_H}
)
add_executable(timelapsesample ${SOURCE})
target_include_directories (timelapsesample PRIVATE
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
  timelapsesample
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
  ZXing::Core
  ZXing::ZXing	
  BZip2::BZip2
  ZLIB::ZLIB
  liblzma::liblzma
  bigint::bigint
  sfml-audio
  whisper::whisper
)