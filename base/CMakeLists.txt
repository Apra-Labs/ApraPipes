cmake_minimum_required(VERSION 3.14)
OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" OFF)
OPTION(ENABLE_WINDOWS "Use this switch to enable WINDOWS" OFF)

SET( CONFIG_FILEPATH ${CMAKE_SOURCE_DIR}/../cmake/config.cmake)

include("../cmake/HunterGate.cmake")
HunterGate(
	URL "https://github.com/saichaitanyaeruvaram/hunter/archive/opencv-4.5.0.tar.gz"
    SHA1 "1258e48a1c06ce027b6d3aa141c7f9661ff0d843"
	FILEPATH ${CONFIG_FILEPATH}
  )
  
  
message("enable cuda condition ------------------------------------------------------ ")
IF(ENABLE_CUDA)
	add_compile_definitions(APRA_CUDA_ENABLED)
	message("enable cuda is on------------------------------------------------------------------")
ENDIF(ENABLE_CUDA)
message("enable cuda DONE ------------------------------------------------------ ")



project(APRAPIPES)
set(CMAKE_CXX_STANDARD 14)

hunter_add_package(OpenCV)
find_package(OpenCV)






