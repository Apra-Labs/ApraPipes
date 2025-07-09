set(TARGET generateTimeLapse)

set(SAMPLE_FILES
  timelapse_summary.cpp
  PipelineMain.cpp
)
set(SAMPLE_FILES_H
  timelapse_summary.h
)
set(SOURCE
  ${SAMPLE_FILES}
  ${SAMPLE_FILES_H}
)

add_executable(${TARGET} ${SOURCE})

target_include_directories(${TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../base/include                  
  ${JETSON_MULTIMEDIA_LIB_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${LIBMP4_INC_DIR}
)

target_include_directories(${TARGET} PUBLIC
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
  ${LIBMP4_LIB}
  ${OPENH264_LIB}
  ${Boost_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
)


message(STATUS "Includes for target '${TARGET}':")
get_target_property(include_dirs ${TARGET} INCLUDE_DIRECTORIES)
message(STATUS "${include_dirs}")


