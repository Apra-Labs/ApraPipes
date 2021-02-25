cmake_minimum_required(VERSION 3.14)

SET( CONFIG_FILEPATH ${CMAKE_SOURCE_DIR}/../cmake/config.cmake)

include("../cmake/HunterGate.cmake")
HunterGate(
	URL "https://github.com/saichaitanyaeruvaram/hunter/archive/opencv-4.5.0.tar.gz"
        SHA1 "1258e48a1c06ce027b6d3aa141c7f9661ff0d843"
	FILEPATH ${CONFIG_FILEPATH}
  )
  
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc") ///  


IF(ENABLE_LINUX)
	add_compile_definitions(LINUX)	
ENDIF(ENABLE_LINUX)



project(APRAPIPES)
set(CMAKE_CXX_STANDARD 14)

hunter_add_package(Boost COMPONENTS system thread filesystem serialization log chrono)
find_package(Boost CONFIG REQUIRED system thread filesystem serialization log chrono)

hunter_add_package(OpenCV)
find_package(OpenCV)

