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

IF(ENABLE_ARM64)
	add_compile_definitions(ARM64)	
ENDIF(ENABLE_ARM64)

project(APRAPIPES)
set(CMAKE_CXX_STANDARD 14)

hunter_add_package(Boost COMPONENTS system thread filesystem serialization log chrono)
find_package(Boost CONFIG REQUIRED system thread filesystem serialization log chrono)

hunter_add_package(OpenCV)
find_package(OpenCV)

hunter_add_package(BZip2)
find_package(BZip2 CONFIG REQUIRED)

hunter_add_package(ZLIB)
find_package(ZLIB CONFIG REQUIRED)

hunter_add_package(lzma)
find_package(lzma CONFIG REQUIRED)


IF(NOT ENABLE_ARM64)
	hunter_add_package(ncursesw)
	find_package(ncursesw CONFIG REQUIRED)
	SET(CURSES_INCLUDE_DIR ${NCURSESW_ROOT}/include/ncursesw)
	SET(CURSES_LIBRARIES PkgConfig::ncursesw)
ENDIF(NOT ENABLE_ARM64)

IF(ENABLE_CUDA)
	enable_language(CUDA)
ENDIF(ENABLE_CUDA)

SET(FFMPEG_ROOT ../thirdparty/ffmpeg)
find_library(LIB_AVFORMAT libavformat.a PATHS ${FFMPEG_ROOT}/libavformat NO_DEFAULT_PATH)
find_library(LIB_AVCODEC libavcodec.a PATHS ${FFMPEG_ROOT}/libavcodec NO_DEFAULT_PATH)
find_library(LIB_AVUTIL libavutil.a PATHS ${FFMPEG_ROOT}/libavutil NO_DEFAULT_PATH)
find_library(LIB_FFSWR libswresample.a PATHS ${FFMPEG_ROOT}/libswresample NO_DEFAULT_PATH)
find_library(LIB_ZXING libZXing.a PATHS ../thirdparty/zxing-cpp/_install/lib NO_DEFAULT_PATH)


IF(ENABLE_CUDA)
	IF(ENABLE_ARM64)
		SET(NVCODEC_LIB)
		SET ( JETSON_MULTIMEDIA_LIB_INCLUDE "/usr/src/jetson_multimedia_api/include" )					

		list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra)
		list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra-egl)

		find_library(NVJPEGLIB_L4T nvjpeg)
		find_library(V4L2LIB nvv4l2)
		find_library(EGLLIB EGL)
		find_library(GLESv2LIB GLESv2)
		find_library(X11LIB X11)
		find_library(NVBUFUTILSLIB nvbuf_utils)
		find_library(EGLSTREAM_CAMCONSUMER_LIB nveglstream_camconsumer)
		find_library(NVARGUS_SOCKETCLINET_LIB nvargus_socketclient)
		find_package(Curses REQUIRED)

		SET(JETSON_LIBS
			libcudart_static.a	
			libcuda.so.1.1	
			${V4L2LIB}
			${NVBUFUTILSLIB}
			${EGLLIB}
			${GLESv2LIB}
			${X11LIB}	
			${EGLSTREAM_CAMCONSUMER_LIB}
			${NVARGUS_SOCKETCLINET_LIB}
		)
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
	../thirdparty/ffmpeg/include
	${CURSES_INCLUDE_DIR}
	../thirdparty/zxing-cpp/_install/include/ZXing
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
	src/FIndexStrategy.cpp
	src/KeyboardListener.cpp
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
	include/FIndexStrategy.h
	include/KeyboardListener.h
)

IF(ENABLE_WINDOWS)
	SET(CORE_FILES_H ${CORE_FILES_H}
		include/targetver.h
		include/stdafx.h
	)
ENDIF(ENABLE_WINDOWS)

SET(GENERIC_FILES
	src/VirtualCameraSink.cpp
	src/RTSPPusher.cpp
	src/H264FrameDemuxer.cpp
	src/H264ParserUtils.cpp
	src/H264Utils.cpp
	src/QRReader.cpp
)
SET(GENERIC_FILES_H
	include/VirtualCameraSink.h
	include/H264FrameDemuxer.h
	include/H264ParserUtils.h
	include/H264Utils.h
	include/RTSPPusher.h
	include/QRReader.h
)

SET(IP_FILES
	src/ApraLines.cpp
	src/CalcHistogramCV.cpp
	src/HistogramOverlay.cpp
	src/ImageDecoderCV.cpp
	src/ImageViewerModule.cpp
	src/BMPConverter.cpp
	src/ImageResizeCV.cpp
	src/ImageEncoderCV.cpp
	src/RotateCV.cpp
	src/BrightnessContrastControl.cpp
	src/VirtualPTZ.cpp
	src/WebCamSource.cpp
)

SET(IP_FILES_H
	include/HistogramOverlay.h
	include/CalcHistogramCV.h
	include/ApraPoint2f.h
	include/ApraLines.h
	include/ImageViewerModule.h
	include/ImageDecoderCV.h
	include/BMPConverter.h
	include/ImageResizeCV.h
	include/ImageEncoderCV.h
	include/RotateCV.h
	include/BrightnessContrastControl.h
	include/VirtualPTZ.h
	include/WebCamSource.h
)

SET(CUDA_CORE_FILES
	src/apra_cudamalloc_allocator.cu
	src/apra_cudamallochost_allocator.cu
	src/CudaCommon.cpp
	src/CudaMemCopy.cpp
	src/CudaStreamSynchronize.cpp
	src/CuCtxSynchronize.cpp
)

SET(CUDA_CORE_FILES_H
	include/CudaStreamSynchronize.h
	include/CudaMemCopy.h
	include/CudaCommon.h	
	include/apra_cudamallochost_allocator.h
	include/apra_cudamalloc_allocator.h
	include/CuCtxSynchronize.h
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
	src/RotateNPPI.cpp
)

IF(ENABLE_ARM64)
	SET(CUDA_IP_FILES ${CUDA_IP_FILES}
		src/JPEGDecoderL4TM.cpp
		src/JPEGDecoderL4TMHelper.cpp
		src/JPEGEncoderL4TM.cpp
		src/JPEGEncoderL4TMHelper.cpp
		src/AV4L2Buffer.cpp
		src/AV4L2ElementPlane.cpp
		src/H264EncoderV4L2Helper.cpp
		src/V4L2CUYUV420Converter.cpp
		src/H264EncoderV4L2.cpp
		src/DMAFDWrapper.cpp
		src/NvArgusCameraHelper.cpp
		src/NvArgusCamera.cpp
		src/NvV4L2Camera.cpp
		src/NvV4L2CameraHelper.cpp
		src/EglRenderer.cpp
		src/NvEglRenderer.cpp
		src/DMAUtils.cpp
		src/NvTransform.cpp
		src/ApraEGLDisplay.cpp
		src/DMAFDToHostCopy.cpp
		src/AffineTransform.cpp
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
	include/RotateNPPI.h
)

IF(ENABLE_ARM64)
	SET(CUDA_IP_FILES_H ${CUDA_IP_FILES_H}
		include/JPEGDecoderL4TMHelper.h
		include/JPEGDecoderL4TM.h	 
		include/JPEGEncoderL4TMHelper.h
		include/JPEGEncoderL4TM.h
		include/AV4L2Buffer.h
		include/AV4L2ElementPlane.h
		include/H264EncoderV4L2Helper.h
		include/V4L2CUYUV420Converter.h
		include/H264EncoderV4L2.h
		include/DMAAllocator.h
		include/DMAFDWrapper.h
		include/Allocators.h
		include/NvArgusCameraHelper.h
		include/NvArgusCamera.h
		include/NvV4L2Camera.h
		include/NvV4L2CameraHelper.h
		include/EglRenderer.h
		include/ApraNvEglRenderer.h
		include/DMAUtils.h
		include/NvTransform.h
		include/ApraEGLDisplay.h
		include/DMAFrameUtils.h
		include/DMAFDToHostCopy.h
		include/AffineTransform.h
	)
ELSE()
	SET(CUDA_IP_FILES_H ${CUDA_IP_FILES_H}		
		include/JPEGEncoderNVJPEG.h	
		include/JPEGDecoderNVJPEG.h
	)
ENDIF(ENABLE_ARM64)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SOURCE
	${CORE_FILES} ${CORE_FILES_H}
	${GENERIC_FILES} ${GENERIC_FILES_H}
	${IP_FILES} ${IP_FILES_H}
)

IF(ENABLE_CUDA)
	set(SOURCE ${SOURCE}
		${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
		${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
	)
ENDIF(ENABLE_CUDA)

add_library(aprapipes STATIC ${SOURCE})

target_include_directories ( aprapipes PRIVATE ${JETSON_MULTIMEDIA_LIB_INCLUDE} ${FFMPEG_ROOT})

# aprapipes Unit Tests

IF (ENABLE_ARM64)
	SET(ARM64_UT_FILES
		test/jpegencoderl4tm_tests.cpp
		test/jpegdecoderl4tm_tests.cpp
		test/opencvresize_tests.cpp
		test/h264encoderv4l2helper_tests.cpp
		test/h264encoderv4l2_tests.cpp
		test/nvarguscamerahelper_tests.cpp
		test/nvarguscamera_tests.cpp
		test/nvv4l2camera_test.cpp
		test/nvv4l2camerahelper_test.cpp
		test/nvtransform_tests.cpp
		test/eglrenderer_test.cpp
	)
ENDIF(ENABLE_ARM64)

IF (ENABLE_CUDA)
	SET(CUDA_UT_FILES
		test/cudamemcopy_tests.cpp		
		test/resizenppi_tests.cpp
		test/rotatenppi_tests.cpp
	)

	IF(ENABLE_ARM64)
		SET(CUDA_UT_FILES ${CUDA_UT_FILES}
			test/cuda_arm_tests.cpp
			test/apraegldisplay_tests.cpp
			test/frame_factory_test_dma.cpp
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
	test/bmpconverter_tests.cpp
	test/virtualcamerasink_tests.cpp
	test/rtsppusher_tests.cpp
	test/findexstrategy_tests.cpp	
	test/jpegdecodercv_tests.cpp
	test/Imageresizecv_tests.cpp
	test/ImageEncodeCV_tests.cpp
	test/QRReader_tests.cpp
	test/rotatecv_tests.cpp
	test/brightness_contrast_tests.cpp
	test/virtualptz_tests.cpp
	test/webcam_source_tests.cpp
	${ARM64_UT_FILES}
	${CUDA_UT_FILES}	
)

add_executable(aprapipesut ${UT_FILES})

IF(ENABLE_ARM64)
	target_include_directories ( aprapipesut PRIVATE ${JETSON_MULTIMEDIA_LIB_INCLUDE} ${FFMPEG_ROOT})
ENDIF(ENABLE_ARM64)

target_link_libraries(aprapipesut   
  aprapipes  
  Boost::filesystem
  Boost::serialization
  Boost::log  
  Boost::thread  
  Boost::chrono
  Boost::system
  Boost::boost    
  ${JETSON_LIBS}
  ${OpenCV_LIBS}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
  ${NVJPEGLIB_L4T}
  ${LIB_AVFORMAT}
  ${LIB_AVCODEC}
  ${LIB_AVUTIL}
  ${LIB_FFSWR}
  ${CURSES_LIBRARIES}
  ${LIB_ZXING}
  BZip2::bz2
  ZLIB::zlib
  lzma::lzma
)
