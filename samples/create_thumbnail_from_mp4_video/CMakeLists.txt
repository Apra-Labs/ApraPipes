set(TARGET create_thumbnail_from_mp4_video)
SET(CORE_FILES
	generate_thumbnail_from_mp4_video.cpp
)
SET(CORE_FILES_H
	GenerateThumbnailsPipeline.h
)
SET(SOURCE
  ${CORE_FILES}
  ${CORE_FILES_H}
)
set(UT_FILE
	test_generate_thumbnail_from_mp4_video.cpp
)
add_library(create_thumbnail_from_mp4_video_source STATIC ${SOURCE})
add_executable(${TARGET} ${UT_FILE})
target_include_directories ( create_thumbnail_from_mp4_video_source PRIVATE
  ${JETSON_MULTIMEDIA_LIB_INCLUDE}
  ${FFMPEG_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
  ${BARESIP_INC_DIR}
  ${LIBRE_INC_DIR}
  ${NVCODEC_INCLUDE_DIR}
)
target_include_directories ( ${TARGET} PUBLIC
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
  create_thumbnail_from_mp4_video_source
  )