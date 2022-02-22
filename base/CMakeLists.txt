cmake_minimum_required(VERSION 3.14)

OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" OFF)
OPTION(ENABLE_WINDOWS "Use this switch to enable WINDOWS" OFF)

SET(PKG_CONFIG_PATH ${CMAKE_SOURCE_DIR}/../thirdparty/gstreamer/outInstall/lib/pkgconfig)
message(STATUS ${PKG_CONFIG_PATH}">>>>>> PKG_CONFIG_PATH")
set(VCPKG_INSTALL_OPTIONS "--clean-after-build")
IF(ENABLE_CUDA)
	add_compile_definitions(APRA_CUDA_ENABLED)
ENDIF(ENABLE_CUDA)

IF(ENABLE_LINUX)
	add_compile_definitions(LINUX)	
ENDIF(ENABLE_LINUX)

IF(ENABLE_WINDOWS)
	add_compile_definitions(WINDOWS)
	set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "x64-windows")
ENDIF(ENABLE_WINDOWS)

IF(ENABLE_ARM64)
	add_compile_definitions(ARM64)
	set(VCPKG_OVERLAY_PORTS ../vcpkg/ports/cudnn)	
	set(VCPKG_OVERLAY_TRIPLETS ../vcpkg/triplets/community/arm64-linux.cmake)
ENDIF(ENABLE_ARM64)

project(APRAPIPES)
set(CMAKE_CXX_STANDARD 14)

IF(ENABLE_LINUX)
	set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/../thirdparty/gst-build/gst-build-1.16/outInstall/lib/x86_64-linux-gnu/pkgconfig/")
ENDIF(ENABLE_LINUX)

IF(ENABLE_WINDOWS)
	set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/../thirdparty/gst-build/gst-build-1.16/outInstall/lib/pkgconfig")
ENDIF(ENABLE_WINDOWS)

find_package(PkgConfig REQUIRED)

pkg_search_module(GSTREAMER REQUIRED gstreamer-1.0)  
pkg_search_module(GOBJECT REQUIRED gobject-2.0)  
pkg_search_module(GLIB REQUIRED glib-2.0)  
pkg_search_module(GIO REQUIRED gio-2.0) 
pkg_search_module(GSTRTP REQUIRED gstreamer-rtp-1.0)  
pkg_search_module(GSTRTSP REQUIRED gstreamer-rtsp-1.0)  
pkg_search_module(GSTRTSPSERVER REQUIRED gstreamer-rtsp-server-1.0)  
pkg_search_module(GSTAPP REQUIRED gstreamer-app-1.0)
pkg_search_module(GSTNET REQUIRED gstreamer-net-1.0)
pkg_search_module(GSTWEBRTC REQUIRED gstreamer-webrtc-1.0)
pkg_search_module(PCRE REQUIRED libpcre)
pkg_search_module(GTHREAD REQUIRED gthread-2.0)
pkg_search_module(GMODULE REQUIRED gmodule-2.0)

if(!ENABLE_WINDOWS)
	pkg_search_module(LIBMOUNT REQUIRED libmount)
	pkg_search_module(LIBBBLKID REQUIRED libblkid)
	pkg_search_module(LIBSELINUX REQUIRED libselinux)

endif()
pkg_search_module(GSTBASE REQUIRED gstreamer-base-1.0)

find_package(Boost COMPONENTS system thread filesystem serialization log chrono unit_test_framework REQUIRED)

find_package(OpenCV CONFIG REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(liblzma REQUIRED)
find_package(FFMPEG REQUIRED)
find_package(ZXing CONFIG REQUIRED)
find_package(bigint CONFIG REQUIRED)
find_package(SFML COMPONENTS system window audio graphics CONFIG REQUIRED)

IF(NOT ENABLE_ARM64)
	# hunter_add_package(ncursesw)
	# find_package(ncursesw CONFIG REQUIRED)
	# SET(CURSES_INCLUDE_DIR ${NCURSESW_ROOT}/include/ncursesw)
	# SET(CURSES_LIBRARIES PkgConfig::ncursesw)
ENDIF(NOT ENABLE_ARM64)

IF(ENABLE_CUDA)
	enable_language(CUDA)
ENDIF(ENABLE_CUDA)

IF(ENABLE_CUDA)
	IF(ENABLE_ARM64)
		SET(NVCODEC_LIB)
		SET(JETSON_MULTIMEDIA_LIB_INCLUDE "/usr/src/jetson_multimedia_api/include")					

		list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra)
		list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra-egl)

		find_library(NVJPEGLIB_L4T nvjpeg REQUIRED)
		find_library(V4L2LIB nvv4l2 REQUIRED)
		find_library(EGLLIB EGL REQUIRED)
		find_library(GLESv2LIB GLESv2 REQUIRED)
		find_library(X11LIB X11 REQUIRED)
		find_library(NVBUFUTILSLIB nvbuf_utils REQUIRED)
		find_library(EGLSTREAM_CAMCONSUMER_LIB nveglstream_camconsumer REQUIRED)
		find_library(NVARGUS_SOCKETCLINET_LIB nvargus_socketclient REQUIRED)
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
		include_directories(AFTER SYSTEM /usr/local/cuda-10.2/include)
	ELSEIF(ENABLE_LINUX)
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
		include_directories(AFTER SYSTEM /usr/local/cuda-10.2/include)
	ELSE()
		find_library(LIBNVCUVID nvcuvid.lib PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH) #change for windows
		find_library(LIBNVENCODE nvencodeapi.lib PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH) #change for windows
		SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})

		SET(NVCUDAToolkit_LIBS		
			cudart_static.lib
			nvjpeg.lib
			nppig.lib
			nppicc.lib
			nppidei.lib
			nppial.lib					
			cublas.lib	
			cublasLt.lib
		)
		include_directories(AFTER SYSTEM "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")
	ENDIF(ENABLE_ARM64)

ENDIF(ENABLE_CUDA)

include_directories(AFTER SYSTEM include)

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
	src/AudioCaptureSrc.cpp
	src/QRReader.cpp
)

SET(CORE_FILES_H
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
	include/AudioCaptureSrc.h
	include/QRReader.h
)

IF(ENABLE_WINDOWS)
	SET(CORE_FILES_H ${CORE_FILES_H}
		include/targetver.h
		include/stdafx.h
	)
ENDIF(ENABLE_WINDOWS)

SET(GENERIC_FILES
	src/RTSPPusher.cpp
	src/H264FrameDemuxer.cpp
	src/H264ParserUtils.cpp
	src/H264Utils.cpp
	src/QRReader.cpp
)
SET(GENERIC_FILES_H
	include/H264FrameDemuxer.h
	include/H264ParserUtils.h
	include/H264Utils.h
	include/RTSPPusher.h
	include/QRReader.h
)

IF(ENABLE_LINUX)
	list(APPEND CORE_FILES src/KeyboardListener.cpp)
	list(APPEND CORE_FILES_H include/KeyboardListener.h)
	list(APPEND GENERIC_FILES src/VirtualCameraSink.cpp)
	list(APPEND GENERIC_FILES_H include/VirtualCameraSink.h)
ENDIF(ENABLE_LINUX)

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
	src/GstOnvifRtspSink.cpp
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
	include/GstOnvifRtspSink.h
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
message(STATUS "-------------Printing Include folder-----------------${GSTREAMER_INCLUDE_DIRS}")
target_include_directories ( aprapipes PRIVATE 
${JETSON_MULTIMEDIA_LIB_INCLUDE}
${FFMPEG_INCLUDE_DIRS} 
${OpenCV_INCLUDE_DIRS} 
${Boost_INCLUDE_DIRS}
${GSTAPP_INCLUDE_DIRS}
${GSTBASE_INCLUDE_DIRS}
${GSTREAMER_INCLUDE_DIRS}
${GSTRTSPSERVER_INCLUDE_DIRS}
${GSTWEBRTC_INCLUDE_DIRS}
${PCRE_INCLUDE_DIRS}
${GLIB_INCLUDE_DIRS}
${GSTRTSP_INCLUDE_DIRS}
${GSTRTP_INCLUDE_DIRS}
${GSTSDP_INCLUDE_DIRS}
${GSTNET_INCLUDE_DIRS}
${GSTPBUTILS_INCLUDE_DIRS}
${GSTRTPMANAGER_INCLUDE_DIRS}
${LIBFFI_INCLUDE_DIRS}
${LIBBLKID_INCLUDE_DIRS}
${GIO_INCLUDE_DIRS}
${GMODULE_INCLUDE_DIRS}
${GTHREAD_INCLUDE_DIRS}
${GOBJECT_INCLUDE_DIRS}
${LIBMOUNT_INCLUDE_DIRS}
${LIBSELINUX_INCLUDE_DIRS}
)

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
	test/rtsppusher_tests.cpp
	test/findexstrategy_tests.cpp	
	test/jpegdecodercv_tests.cpp
	test/Imageresizecv_tests.cpp
	test/ImageEncodeCV_tests.cpp
	test/rotatecv_tests.cpp
	test/brightness_contrast_tests.cpp
	test/virtualptz_tests.cpp
	test/webcam_source_tests.cpp
	test/sound_record_tests.cpp
	test/QRReader_tests.cpp
	test/gstrtsponvifsink_tests.cpp
	${ARM64_UT_FILES}
	${CUDA_UT_FILES}
)

IF(ENABLE_LINUX)
	list(APPEND UT_FILES
	test/virtualcamerasink_tests.cpp
	test/QRReader_tests.cpp
	)
	SET(LIBRESOLV resolv)
ENDIF(ENABLE_LINUX)

add_executable(aprapipesut ${UT_FILES})

IF(ENABLE_ARM64)
	target_include_directories ( aprapipesut PRIVATE ${JETSON_MULTIMEDIA_LIB_INCLUDE} ${FFMPEG_ROOT})
ENDIF(ENABLE_ARM64)

#add_library(gstreamer_dll SHARED 
# )
message(STATUS ${GSTRTSP_LIBRARIES}"RTSP LIBRARIES >>>>")

target_link_libraries(aprapipesut   
  aprapipes  
  ${Boost_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${JETSON_LIBS}
  ${NVCUDAToolkit_LIBS}
  ${NVCODEC_LIB}
  ${NVJPEGLIB_L4T}
  ${CURSES_LIBRARIES}
  ${GSTAPP_LIBRARIES}
  ${GSTBASE_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTWEBRTC_LIBRARIES}
  ${GSTRTSPSERVER_LIBRARIES}
  ${PCRE_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${GSTRTSP_LIBRARIES}
  ${GSTRTP_LIBRARIES}
  ${GSTSDP_LIBRARIES}
  ${GSTNET_LIBRARIES}
  ${GSTPBUTILS_LIBRARIES}
  ${GSTRTPMANAGER_LIBRARIES}
  ${LIBFFI_LIBRARIES}
  ${LIBBLKID_LIBRARIES}
  ${GIO_LIBRARIES}
  ${GMODULE_LIBRARIES}
  ${GTHREAD_LIBRARIES}
  ${GLIB_LDFLAGS}
  ${GOBJECT_LIBRARIES}
  ${LIBMOUNT_LIBRARIES}
  ${LIBSELINUX_LIBRARIES}

  ${GSTAPP_LIBRARIES}
  ${GSTBASE_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTWEBRTC_LIBRARIES}
  ${GSTRTSPSERVER_LIBRARIES}
  ${PCRE_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${GSTRTSP_LIBRARIES}
  ${GSTRTP_LIBRARIES}
  ${GSTSDP_LIBRARIES}
  ${GSTNET_LIBRARIES}
  ${GSTPBUTILS_LIBRARIES}
  ${GSTRTPMANAGER_LIBRARIES}
  ${LIBFFI_LIBRARIES}
  ${LIBBLKID_LIBRARIES}
  ${GIO_LIBRARIES}
  ${GMODULE_LIBRARIES}
  ${GTHREAD_LIBRARIES}
  ${GLIB_LDFLAGS}
  ${GOBJECT_LIBRARIES}
  ${LIBMOUNT_LIBRARIES}
  ${LIBSELINUX_LIBRARIES}
    
  ${GSTAPP_LIBRARIES}
  ${GSTBASE_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTRTSPSERVER_LIBRARIES}
  ${PCRE_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${GSTRTSP_LIBRARIES}
  ${GSTRTP_LIBRARIES}
  ${GSTSDP_LIBRARIES}
  ${GSTNET_LIBRARIES}
  ${GSTPBUTILS_LIBRARIES}
  ${GSTRTPMANAGER_LIBRARIES}
  ${LIBFFI_LIBRARIES}
  ${LIBBLKID_LIBRARIES}
  ${GIO_LIBRARIES}
  ${GMODULE_LIBRARIES}
  ${GTHREAD_LIBRARIES}
  ${GLIB_LDFLAGS}
  ${GOBJECT_LIBRARIES}
  ${LIBMOUNT_LIBRARIES}
  ${LIBSELINUX_LIBRARIES}
  ${LIBRESOLV}
  ZXing::Core
  ZXing::ZXing	
  BZip2::BZip2
  ZLIB::ZLIB
  liblzma::liblzma
  bigint::bigint
  sfml-audio
  )

#add_custom_command(TARGET aprapipesut
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gstreamer_dll> "${CMAKE_CURRENT_BINARY_DIR}"
#    COMMENT "Copy dll file to ${CMAKE_CURRENT_BINARY_DIR} directory" VERBATIM
#)