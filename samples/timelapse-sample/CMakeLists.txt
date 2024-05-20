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
set(UT_FILE
	timelapse_summary_test.cpp
)
find_library(APRA_PIPES_LIB NAMES aprapipes)
add_library(example_aprapipes STATIC ${SOURCE})
add_executable(runner_test ${UT_FILE})
target_include_directories ( example_aprapipes PRIVATE
  ${JETSON_MULTIMEDIA_LIB_INCLUDE}
  ${FFMPEG_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
  ${BARESIP_INC_DIR}
  ${LIBRE_INC_DIR}
  ${NVCODEC_INCLUDE_DIR}
)
target_include_directories ( runner_test PUBLIC
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
  runner_test
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
  example_aprapipes
  )