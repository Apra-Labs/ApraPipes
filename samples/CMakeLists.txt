#dependencies

find_package(Threads REQUIRED)

include_directories(
../base
${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(timelapse-sample)