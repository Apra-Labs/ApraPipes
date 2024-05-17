#dependencies

find_package(Threads REQUIRED)

include_directories(
../base
${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(create_thumbnail_from_mp4_video)
add_subdirectory(play_mp4_from_begining)
