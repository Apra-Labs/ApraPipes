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

IF(ENABLE_CUDA)
	add_compile_definitions(APRA_CUDA_ENABLED)
ENDIF(ENABLE_CUDA)

IF(ENABLE_LINUX)
	add_compile_definitions(LINUX)	
ENDIF(ENABLE_LINUX)

list(APPEND CMAKE_PREFIX_PATH "/usr/lib/aarch64-linux-gnu/tegra")

project(APRAPIPES)
hunter_add_package(Boost COMPONENTS system thread filesystem serialization log chrono)
find_package(Boost CONFIG REQUIRED system thread filesystem serialization log chrono)

hunter_add_package(OpenCV)
find_package(OpenCV)

enable_language(CUDA)

IF(ENABLE_CUDA)
	IF(ENABLE_ARM64)
		SET(NVCODEC_LIB)
		SET ( JPEGLIBINCLUDE "/usr/src/jetson_multimedia_api/include" )			
		find_library(NVJPEGLIB_L4T nvjpeg)
	ELSE()
		find_library(LIBNVCUVID libnvcuvid.so PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64 NO_DEFAULT_PATH)
		find_library(LIBNVENCODE libnvidia-encode.so PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64 NO_DEFAULT_PATH)
		SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})
		
		SET(NVCUDAToolkit_LIBS		
			libcudart_static.a
			libnvjpeg_static.a
			libnppig_static.a
			libnppicc_static.a
			libnppidei_static.a
			libnppial_static.a					
			libculibos.a	
		)	

	ENDIF(ENABLE_ARM64)
ENDIF(ENABLE_CUDA)

include_directories(AFTER SYSTEM include
	${BOOST_ROOT}/include
	${OPENCV_ROOT}/include/opencv4
	/usr/local/cuda-10.2/include
	../thirdparty/Video_Codec_SDK_10.0.26/Interface
)

# ApraPipes library

SET(CORE_FILES
	src/ApraPool.cpp
	src/FilenameStrategy.cpp
	src/FileReaderModule.cpp
	src/FileSequenceDriver.cpp
	src/FileWriterModule.cpp
	src/FrameContainerQueue.cpp
	src/Frame.cpp
	src/FrameFactory.cpp
	src/FramesMuxer.cpp
	src/Logger.cpp
	src/Merge.cpp
	src/Module.cpp
	src/PipeLine.cpp
	src/QuePushStrategy.cpp
	src/Split.cpp
	src/Utils.cpp
)

SET(include/CORE_FILES_H
	include/FramesMuxer.h
	include/FrameMetadata.h
	include/FrameMetadataFactory.h
	include/Frame.h
	include/FrameFactory.h
	include/FrameContainerQueue.h
	include/FileWriterModule.h
	include/FileSequenceDriver.h
	include/FileReaderModule.h
	include/FilenameStrategy.h
	include/ExtFrame.h
	include/ExternalSourceModule.h
	include/ExternalSinkModule.h
	include/CommonDefs.h
	include/Command.h
	include/BoundBuffer.h
	include/ArrayMetadata.h
	include/ApraPool.h
	include/ApraData.h
	include/AIPExceptions.h
	include/Utils.h
	include/ThreadSafeQue.h	
	include/StatSink.h
	include/Split.h
	include/ROIMetadata.h
	include/RawImagePlanarMetadata.h
	include/RawImageMetadata.h
	include/QuePushStrategy.h
	include/PipeLine.h
	include/PausePlayMetadata.h
	include/PaceMaker.h
	include/Module.h
	include/Merge.h
	include/Logger.h
	include/LineMetadata.h
	include/ImageMetadata.h
	include/enum_macros.h
	include/MetadataHints.h
)

IF(ENABLE_WINDOWS)
	SET(CORE_FILES_H ${CORE_FILES_H}
		include/targetver.h
		include/stdafx.h
	)
ENDIF(ENABLE_WINDOWS)

SET(GENRIC_FILES)
SET(GENRIC_FILES_H)

SET(IP_FILES
	src/ApraLines.cpp
	src/CalcHistogramCV.cpp
	src/HistogramOverlay.cpp
	src/ImageDecoderCV.cpp
	src/ImageViewerModule.cpp
)

SET(IP_FILES_H
	include/HistogramOverlay.h
	include/CalcHistogramCV.h
	include/ApraPoint2f.h
	include/ApraLines.h
	include/ImageViewerModule.h
	include/ImageDecoderCV.h
)

SET(CUDA_CORE_FILES
	src/apra_cudamalloc_allocator.cu
	src/apra_cudamallochost_allocator.cu
	src/CudaCommon.cpp
	src/CudaMemCopy.cpp
	src/CudaStreamSynchronize.cpp
)

SET(CUDA_CORE_FILES_H
	include/CudaStreamSynchronize.h
	include/CudaMemCopy.h
	include/CudaCommon.h	
	include/apra_cudamallochost_allocator.h
	include/apra_cudamalloc_allocator.h
)

SET(CUDA_IP_FILES
	src/build_point_list.cu
	src/CCKernel.cu
	src/CCNPPI.cpp
	src/EffectsKernel.cu
	src/EffectsNPPI.cpp
	src/GaussianBlur.cpp
	src/OverlayKernel.cu
	src/OverlayNPPI.cpp
	src/ResizeNPPI.cpp
)

IF(ENABLE_ARM64)
	SET(CUDA_IP_FILES ${CUDA_IP_FILES}
		src/JPEGDecoderL4TM.cpp
		src/JPEGDecoderL4TMHelper.cpp
		src/JPEGEncoderL4TM.cpp
		src/JPEGEncoderL4TMHelper.cpp
	)
ELSE()
	SET(CUDA_IP_FILES ${CUDA_IP_FILES}		
		src/JPEGDecoderNVJPEG.cpp	
		src/JPEGEncoderNVJPEG.cpp	
	)
ENDIF(ENABLE_ARM64)

SET(CUDA_IP_FILES_H
	include/GaussianBlur.h
	include/EffectsNPPI.h
	include/EffectsKernel.h
	include/CCNPPI.h
	include/CCKernel.h
	include/ResizeNPPI.h
	include/OverlayNPPI.h
	include/OverlayKernel.h
)

IF(ENABLE_ARM64)
	SET(CUDA_IP_FILES_H ${CUDA_IP_FILES_H}
		include/JPEGDecoderL4TMHelper.h
		include/JPEGDecoderL4TM.h	
		include/JPEGEncoderL4TMHelper.h
		include/JPEGEncoderL4TM.h
	)
ELSE()
	SET(CUDA_IP_FILES_H ${CUDA_IP_FILES_H}		
		include/JPEGEncoderNVJPEG.h	
		include/JPEGDecoderNVJPEG.h
	)
ENDIF(ENABLE_ARM64)

add_library(aprapipes STATIC
	${CORE_FILES} ${CORE_FILES_H}
	${GENRIC_FILES} ${GENERIC_FILES_H}
	${IP_FILES} ${IP_FILES_H}
	${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
	${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
)

target_include_directories ( aprapipes PRIVATE ${JPEGLIBINCLUDE})

# aprapipes Unit Tests

IF (ENABLE_ARM64)
	SET(ARM64_UT_FILES
		test/jpegencoderl4tm_tests.cpp
		test/jpegdecoderl4tm_tests.cpp
		test/opencvresize_tests.cpp
	)
ENDIF(ENABLE_ARM64)

IF (ENABLE_CUDA)
	SET(CUDA_UT_FILES
		test/cudamemcopy_tests.cpp		
		test/resizenppi_tests.cpp
	)

	IF(ENABLE_ARM64)
		SET(CUDA_UT_FILES ${CUDA_UT_FILES}

		)
	ELSE()
		SET(CUDA_UT_FILES ${CUDA_UT_FILES}	
			test/jpegencodernvjpeg_tests.cpp
			test/jpegdecodernvjpeg_tests.cpp			
			test/resizenppi_jpegencodernvjpeg_tests.cpp
			test/nvjpeg_combo_tests.cpp			
			test/ccnppi_tests.cpp
			test/overlaynppi_tests.cpp
			test/effectsnppi_tests.cpp
		)
	ENDIF(ENABLE_ARM64)
ENDIF(ENABLE_CUDA)

SET(UT_FILES
	test/utmain.cpp
	test/unit_tests.cpp
	test/cv_memory_leaks_tests.cpp
	test/module_tests.cpp
	test/calchistogramcv_tests.cpp
	test/filenamestrategy_tests.cpp
	test/test_utils.cpp
	test/test_utils.h	
	test/filewritermodule_tests.cpp
	test/logger_tests.cpp
	test/quepushstrategy_tests.cpp	
	test/framesmuxer_tests.cpp
	test/filereadermodule_tests.cpp
	test/merge_tests.cpp
	test/split_tests.cpp
	test/imagemetadata_tests.cpp
	${ARM64_UT_FILES}
	${CUDA_UT_FILES}	
)

add_executable(aprapipesut ${UT_FILES})

target_link_libraries(aprapipesut   
  aprapipes  
  Boost::filesystem
  Boost::serialization
  Boost::log  
  Boost::thread  
  Boost::chrono
  Boost::system
  Boost::boost    
  ${OpenCV_LIBS}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
  ${NVJPEGLIB_L4T}
)