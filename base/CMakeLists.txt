cmake_minimum_required(VERSION 3.29)

OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" OFF)
OPTION(ENABLE_WINDOWS "Use this switch to enable WINDOWS" OFF)
OPTION(ENABLE_MACOS "Use this switch to enable MACOS" OFF)
OPTION(BUILD_HEADLESS "Build without GTK/GUI dependencies (for headless/server environments)" OFF)

# FFmpeg Assembly Optimizations
# Default OFF: FFmpeg built without hand-written assembly, allowing static linking into shared objects (Node.js addon)
# When ON: FFmpeg built with assembly (MMX/SSE/AVX on x64, NEON on ARM64) for better performance,
#          but Node.js addon will exclude FFmpeg-dependent modules (RTSPClientSrc, RTSPPusher, MotionVectorExtractor, H264ParserUtils)
# See docs/declarative-pipeline/BUILD_OPTIONS.md for details
OPTION(USE_FFMPEG_ASM_OPTIMIZATIONS "Enable FFmpeg hand-written assembly optimizations (excludes FFmpeg modules from Node.js addon)" OFF)

# vcpkg overlay ports are controlled via VCPKG_OVERLAY_PORTS environment variable
# ARM64 builds set this in CI workflow to use arm64-overlay directory
# All other builds use standard vcpkg.json versions

set(VCPKG_INSTALL_OPTIONS "--clean-after-build")
IF(ENABLE_CUDA)
	add_compile_definitions(APRA_HAS_CUDA_HEADERS)
	add_compile_definitions(ENABLE_CUDA)
ENDIF(ENABLE_CUDA)

# NOTE: VCPKG_TARGET_TRIPLET must be set via cmake command line (-DVCPKG_TARGET_TRIPLET=...)
# Setting it here happens TOO LATE - after vcpkg toolchain has already initialized
# See CI workflow files (.github/workflows/) for triplet configuration

IF(ENABLE_LINUX)
	add_compile_definitions(LINUX)
ENDIF(ENABLE_LINUX)

IF(ENABLE_WINDOWS)
	add_compile_definitions(WINDOWS)
	set(VCPKG_PLATFORM_TOOLSET "v143" CACHE STRING "v143" FORCE)
ENDIF(ENABLE_WINDOWS)

IF(ENABLE_MACOS)
	add_compile_definitions(MACOS)
ENDIF(ENABLE_MACOS)

IF(ENABLE_ARM64)
	add_compile_definitions(ARM64)
	# set(VCPKG_OVERLAY_PORTS ../vcpkg/ports/cudnn)
	set(VCPKG_OVERLAY_TRIPLETS ../vcpkg/triplets/community/arm64-linux.cmake)
	set(ENV{VCPKG_FORCE_SYSTEM_BINARIES} 1)
	set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
ENDIF(ENABLE_ARM64)

#use /MP only for language CXX (and not CUDA) and MSVC for both targets

add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>)

set(CMAKE_CXX_STANDARD 17)

project(APRAPIPES)

message(STATUS $ENV{PKG_CONFIG_PATH}">>>>>> PKG_CONFIG_PATH")

find_package(PkgConfig REQUIRED)
find_package(Boost COMPONENTS system filesystem serialization log chrono unit_test_framework REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(LibLZMA REQUIRED)

# libarchive is a transitive dependency of OpenCV highgui - we need to link it
# explicitly with its dependencies (BZ2, ZLIB, LZMA) to fix static link order issues
find_library(LIBARCHIVE_LIB NAMES archive libarchive.a
    PATHS ${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib
    NO_DEFAULT_PATH)
if(LIBARCHIVE_LIB)
    message(STATUS "Found libarchive: ${LIBARCHIVE_LIB}")
endif()
find_package(FFMPEG REQUIRED)
find_package(ZXing CONFIG REQUIRED)
find_package(bigint CONFIG REQUIRED)
find_package(SFML COMPONENTS Network Graphics Window Audio System CONFIG REQUIRED)
find_package(whisper CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)


IF(ENABLE_LINUX)
	find_package(GLEW REQUIRED)
	find_package(glfw3 CONFIG REQUIRED)
	find_package(FreeGLUT CONFIG REQUIRED)
	pkg_check_modules(GIO REQUIRED gio-2.0)
	pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
	pkg_check_modules(GLFW REQUIRED glfw3)
ENDIF()

IF(ENABLE_MACOS)
	find_package(GLEW REQUIRED)
	find_package(glfw3 CONFIG REQUIRED)
	# macOS uses native GLUT (not FreeGLUT)
	find_package(GLUT REQUIRED)
	# Skip GTK3/GDK/GIO/GOBJECT - not used in source code
ENDIF()

IF(ENABLE_ARM64)
	# Prepend vcpkg's pkgconfig path so vcpkg-built libraries (and their dependencies) are found first
	# This ensures consistent library versions when vcpkg builds newer deps than system provides
	set(VCPKG_PKGCONFIG_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/pkgconfig")
	set(ENV{PKG_CONFIG_PATH} "${VCPKG_PKGCONFIG_PATH}:/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig")
ENDIF(ENABLE_ARM64)

IF(ENABLE_LINUX AND NOT ENABLE_ARM64 AND NOT BUILD_HEADLESS)
	# x64 Linux uses vcpkg's GTK
	pkg_check_modules(GDK3 REQUIRED gdk-3.0)
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
	# Curses is needed for KeyboardListener
	find_package(Curses REQUIRED)
ELSEIF(ENABLE_LINUX AND NOT ENABLE_ARM64 AND BUILD_HEADLESS)
	# Headless build - skip GTK/GDK
	message(STATUS "BUILD_HEADLESS=ON: Skipping GTK/GDK dependencies")
	set(GDK3_FOUND FALSE)
	set(GTK3_FOUND FALSE)
ELSEIF(ENABLE_ARM64 AND NOT BUILD_HEADLESS)
	# ARM64/Jetson uses system GTK headers directly (see VCPKG_GTK_INCLUDE_DIRS below)
	# and has version mismatch between vcpkg glib (2.86) and system glib (2.64)
	# We need GTK3 for linking GtkGlRenderer
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
ELSEIF(ENABLE_ARM64 AND BUILD_HEADLESS)
	# Headless ARM64 build - skip GTK/GDK (for server/Node.js environments)
	message(STATUS "BUILD_HEADLESS=ON (ARM64): Skipping GTK/GDK dependencies")
	set(GDK3_FOUND FALSE)
	set(GTK3_FOUND FALSE)
ENDIF()

IF(ENABLE_CUDA)
	if((NOT DEFINED CMAKE_CUDA_ARCHITECTURES) OR (CMAKE_CUDA_ARCHITECTURES STREQUAL ""))
  		set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75)
	endif()

	message ("CUDA ARCHS: ${CMAKE_CUDA_ARCHITECTURES}")
	enable_language(CUDA)
	SET(NVCODEC_INCLUDE_DIR ../thirdparty/Video_Codec_SDK_10.0.26/Interface)
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
		# JetPack 5.0+ uses nvbufsurface instead of nvbuf_utils (JetPack 4.x)
		find_library(NVBUFUTILSLIB NAMES nvbufsurface nvbuf_utils REQUIRED)
		# JetPack 5.0+ requires nvbufsurftransform for buffer transforms
		find_library(NVBUFSURFTRANSFORMLIB nvbufsurftransform)
		# nveglstream_camconsumer is only available in JetPack 4.x, not in JetPack 5.x
		find_library(EGLSTREAM_CAMCONSUMER_LIB nveglstream_camconsumer)
		if(NOT EGLSTREAM_CAMCONSUMER_LIB)
			message(STATUS "nveglstream_camconsumer not found (JetPack 5.x) - EGL rendering will be limited")
		endif()
		find_library(NVARGUS_SOCKETCLINET_LIB nvargus_socketclient REQUIRED)
		find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)
		find_library(BARESIP_LIB NAMES libbaresip.so libbaresip.a baresip REQUIRED)
		find_package(Curses REQUIRED)

		set(VCPKG_GTK_INCLUDE_DIRS
    		/usr/include/gtk-3.0/
    		/usr/include/glib-2.0/
    		/usr/include/pango-1.0/
    		/usr/include/harfbuzz/
    		/usr/include/cairo/
    		/usr/include/atk-1.0/
    		/usr/include/gdk-pixbuf-2.0/
    		/usr/lib/glib-2.0/include/
			/usr/lib/aarch64-linux-gnu/glib-2.0/include/
		)


		SET(JETSON_LIBS
			libcudart_static.a
			libcuda.so.1.1
			${V4L2LIB}
			${NVBUFUTILSLIB}
			${EGLLIB}
			${GLESv2LIB}
			${X11LIB}
			${NVARGUS_SOCKETCLINET_LIB}
		)
		# Add nveglstream_camconsumer only if found (JetPack 4.x only)
		if(EGLSTREAM_CAMCONSUMER_LIB)
			list(APPEND JETSON_LIBS ${EGLSTREAM_CAMCONSUMER_LIB})
		endif()
		# Add nvbufsurftransform if found (JetPack 5.x)
		if(NVBUFSURFTRANSFORMLIB)
			list(APPEND JETSON_LIBS ${NVBUFSURFTRANSFORMLIB})
		endif()
		include_directories(AFTER SYSTEM /usr/local/cuda/include)
		include_directories(AFTER SYSTEM  /usr/local/cuda/samples/common/inc/)
		include_directories(AFTER SYSTEM  /usr/include/)
	    include_directories(AFTER SYSTEM  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ApraGTKUtils/includes/)
	ELSEIF(ENABLE_LINUX)
		# Try to find NVIDIA Video Codec libraries in multiple locations:
		# 1. System driver libraries (on bare metal with NVIDIA drivers)
		# 2. Video Codec SDK stub libraries (in thirdparty - for Docker/development)
		# Note: These are NOT part of CUDA toolkit, they're from Video Codec SDK
		find_library(LIBNVCUVID libnvcuvid.so
			PATHS
				/usr/lib/x86_64-linux-gnu
				../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64
			NO_DEFAULT_PATH
		)
		find_library(LIBNVENCODE libnvidia-encode.so
			PATHS
				/usr/lib/x86_64-linux-gnu
				../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64
			NO_DEFAULT_PATH
		)
		find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)
		find_library(BARESIP_LIB NAMES libbaresip.so libbaresip.a baresip REQUIRED)
		SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})
		SET(NVCUDAToolkit_LIBS
			# libcuda.so - REMOVED: Now loaded dynamically at runtime via dlopen (see CudaDriverLoader)
			libcudart_static.a
			libnvjpeg_static.a
			libnppig_static.a
			libnppicc_static.a
			libnppidei_static.a
			libnppial_static.a
			libculibos.a
		) 
		include_directories(AFTER SYSTEM /usr/local/cuda/include)
	ELSE()
		find_library(LIBNVCUVID nvcuvid.lib PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH) #change for windows
		find_library(LIBNVENCODE nvencodeapi.lib PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH) #change for windows
		SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})
		SET(NVCUDAToolkit_LIBS		
			cudart_static.lib
			nvjpeg.lib
			cuda.lib
			nppig.lib
			nppicc.lib
			nppidei.lib
			nppial.lib					
			cublas.lib	
			cublasLt.lib
		)
		include_directories(AFTER SYSTEM "$ENV{CUDA_PATH}/include")
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
	src/Mp4WriterSink.cpp
	src/Mp4WriterSinkUtils.cpp
	src/MultimediaQueueXform.cpp
	src/Mp4ReaderSource.cpp
	src/OverlayModule.cpp
	src/OrderedCacheOfFiles.cpp
	src/SimpleControlModule.cpp
	src/APErrorObject.cpp
	src/APHealthObject.cpp
	src/declarative/ModuleRegistry.cpp
	src/declarative/PipelineDescription.cpp
	src/declarative/JsonParser.cpp
	src/declarative/ModuleFactory.cpp
	src/declarative/FrameTypeRegistry.cpp
	src/declarative/FrameTypeRegistrations.cpp
	src/declarative/PipelineValidator.cpp
	src/declarative/PipelineAnalyzer.cpp
	src/declarative/ModuleRegistrations.cpp
)

SET(CORE_FILES_H
	include/BufferMaker.h	
	include/Mp4ErrorFrame.h
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
	include/Mp4WriterSink.h
	include/Mp4WriterSinkUtils.h
	include/EncodedImageMetadata.h
	include/PropsChangeMetadata.h
	include/ValveModule.h
	include/ArchiveSpaceManager.h
	include/MultimediaQueueXform.h
	include/H264Metadata.h
	include/Mp4ReaderSource.h
	include/OverlayModule.h
	include/OrderedCacheOfFiles.h
	include/TestSignalGeneratorSrc.h
	include/AbsControlModule.h
	include/SimpleControlModule.h
	include/APErrorObject.h
	include/APCallback.h
	include/APHealthObject.h
	include/declarative/Metadata.h
	include/declarative/ModuleRegistry.h
	include/declarative/PipelineDescription.h
	include/declarative/JsonParser.h
	include/declarative/ParseResult.h
	include/declarative/ModuleFactory.h
	include/declarative/FrameTypeRegistry.h
	include/declarative/PipelineValidator.h
)

IF(ENABLE_WINDOWS)
	SET(CORE_FILES_H ${CORE_FILES_H}
		include/targetver.h
		include/stdafx.h
	)
ENDIF(ENABLE_WINDOWS)

SET(GENERIC_FILES
	src/H264FrameDemuxer.cpp
	src/H264Utils.cpp
	src/QRReader.cpp
)
SET(GENERIC_FILES_H
	include/H264FrameDemuxer.h
	include/H264Utils.h
	include/QRReader.h
)

# ============================================================
# FFmpeg-Dependent Modules
# These modules require FFmpeg and are separated to allow conditional
# exclusion from Node.js addon when USE_FFMPEG_ASM_OPTIMIZATIONS is ON.
# When asm optimizations are enabled, static FFmpeg cannot be linked
# into shared objects due to -fPIC relocation issues.
# ============================================================
SET(FFMPEG_DEPENDENT_SOURCES
	src/RTSPClientSrc.cpp
	src/RTSPPusher.cpp
	src/MotionVectorExtractor.cpp
	src/H264ParserUtils.cpp
)
SET(FFMPEG_DEPENDENT_HEADERS
	include/RTSPClientSrc.h
	include/RTSPPusher.h
	include/MotionVectorExtractor.h
	include/H264ParserUtils.h
)

IF(ENABLE_LINUX)
	# VirtualCameraSink doesn't require GTK
	list(APPEND GENERIC_FILES src/VirtualCameraSink.cpp)
	list(APPEND GENERIC_FILES_H include/VirtualCameraSink.h)

	# KeyboardListener requires Curses - only include if not headless
	IF(NOT BUILD_HEADLESS AND CURSES_FOUND)
		list(APPEND CORE_FILES src/KeyboardListener.cpp)
		list(APPEND CORE_FILES_H include/KeyboardListener.h)
	ENDIF()

	# GTK/GL files only included if GTK3 is found (not in headless mode)
	IF(GTK3_FOUND)
		SET(GTKGL_FILES_H include/Background.h
			include/GLUtils.h
			include/GtkGlRenderer.h
			include/GTKMatrix.h
			include/GTKModel.h
			include/GTKSetup.h
			include/GTKView.h
		)
		SET(GTKGL_FILES_CPP src/Background.cpp
			src/GtkGlRenderer.cpp
			src/GTKMatrix.cpp
			src/GTKModel.cpp
			src/GTKSetup.cpp
			src/GTKView.cpp
			)
	ELSE()
		SET(GTKGL_FILES_H "")
		SET(GTKGL_FILES_CPP "")
	ENDIF()
ENDIF(ENABLE_LINUX)

SET(IP_FILES
	src/ApraLines.cpp
	src/CalcHistogramCV.cpp
	src/HistogramOverlay.cpp
	src/ImageDecoderCV.cpp
	src/ImageViewerModule.cpp
	src/BMPConverter.cpp
	src/ImageResizeCV.cpp
	src/FacialLandmarksCV.cpp
	src/ImageEncoderCV.cpp
	src/RotateCV.cpp
	src/AffineTransform.cpp
	src/BrightnessContrastControlXform.cpp
	src/VirtualPTZ.cpp
	src/WebCamSource.cpp
	src/FaceDetectorXform.cpp
	src/TextOverlayXForm.cpp
	src/ValveModule.cpp
	src/ColorConversionXForm.cpp
	src/AbsColorConversionFactory.cpp
	src/ColorConversionStrategy.h
	src/AbsColorConversionFactory.h
	src/ArchiveSpaceManager.cpp
	src/Overlay.cpp
	src/OverlayFactory.h
	src/OverlayFactory.cpp
	src/TestSignalGeneratorSrc.cpp
	src/AudioToTextXForm.cpp 
	src/AbsControlModule.cpp
	src/ThumbnailListGenerator.cpp
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
	include/FacialLandmarksCV.h
	include/ImageEncoderCV.h
	include/RotateCV.h
	include/AffineTransform.h
	include/BrightnessContrastControlXform.h
	include/VirtualPTZ.h
	include/WebCamSource.h
	include/ApraFaceInfo.h
	include/FaceDetectsInfo.h
	include/FaceDetectorXform.h
	include/TextOverlayXForm.h
	include/ColorConversionXForm.h
	include/Overlay.h
	include/AudioToTextXForm.h
	include/ThumbnailListGenerator.h
)

SET(CUDA_CORE_FILES
	src/apra_cudamalloc_allocator.cu
	src/apra_cudamallochost_allocator.cu
	src/CudaMemCopy.cpp
	src/MemTypeConversion.cpp
	src/CudaStreamSynchronize.cpp
	src/CuCtxSynchronize.cpp
	src/CudaCommon.cpp
	src/CudaDriverLoader.cpp
)

SET(CUDA_CORE_FILES_H
	include/CudaStreamSynchronize.h
	include/CudaMemCopy.h
	include/MemTypeConversion.h
	include/apra_cudamallochost_allocator.h
	include/apra_cudamalloc_allocator.h
	include/CuCtxSynchronize.h
	include/CudaCommon.h

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
	src/H264Decoder.cpp
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
		src/H264DecoderV4L2Helper.cpp
		src/H264DecoderV4L2Helper.h
	)
ELSE()
	SET(CUDA_IP_FILES ${CUDA_IP_FILES}		# following modules and related files do not work on ARM64
		src/JPEGDecoderNVJPEG.cpp	
		src/JPEGEncoderNVJPEG.cpp
		src/H264EncoderNVCodecHelper.cpp
		src/H264EncoderNVCodec.cpp
		src/H264DecoderNvCodecHelper.cpp
		src/H264DecoderNvCodecHelper.h
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
	include/H264Decoder.h
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
	SET(CUDA_IP_FILES_H ${CUDA_IP_FILES_H}		# following modules and related files do not work on ARM64
		include/JPEGEncoderNVJPEG.h	
		include/JPEGDecoderNVJPEG.h
		include/H264EncoderNVCodecHelper.h
		include/H264EncoderNVCodec.h
	)
ENDIF(ENABLE_ARM64)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SOURCE
	${CORE_FILES} ${CORE_FILES_H}
	${GENERIC_FILES} ${GENERIC_FILES_H}
	${FFMPEG_DEPENDENT_SOURCES} ${FFMPEG_DEPENDENT_HEADERS}
	${IP_FILES} ${IP_FILES_H}
)

IF(ENABLE_CUDA)
	set(SOURCE ${SOURCE}
		${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
		${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
	)
ENDIF(ENABLE_CUDA)

IF(ENABLE_LINUX)
    set(SOURCE ${SOURCE} 
	${GTKGL_FILES_H} ${GTKGL_FILES_CPP}
	)
ENDIF(ENABLE_LINUX)

add_library(aprapipes STATIC ${SOURCE})

link_directories(${GTK3_LIBRARY_DIRS})
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
target_include_directories ( aprapipes PRIVATE 
	${JETSON_MULTIMEDIA_LIB_INCLUDE}
	${GTK3_INCLUDE_DIRS}
	${VCPKG_GTK_INCLUDE_DIRS}
	${FFMPEG_INCLUDE_DIRS} 
	${OpenCV_INCLUDE_DIRS} 
	${Boost_INCLUDE_DIRS}
	${LIBMP4_INC_DIR}
	${BARESIP_INC_DIR}
	${LIBRE_INC_DIR}
	${NVCODEC_INCLUDE_DIR}
)


# aprapipes Unit Tests

IF (ENABLE_ARM64)
	SET(ARM64_UT_FILES
		test/jpegencoderl4tm_tests.cpp
		test/jpegdecoderl4tm_tests.cpp
#		test/l4tm_dec_enc_1_tests.cpp  #todo this test needs to be improved to add to jetson suite
		test/opencvresize_tests.cpp
		test/h264encoderv4l2_tests.cpp
		test/nvarguscamerahelper_tests.cpp
		test/nvarguscamera_tests.cpp
		test/nvv4l2camera_test.cpp
		test/nvv4l2camerahelper_test.cpp
		test/nvtransform_tests.cpp
		test/eglrenderer_test.cpp
		test/cuda_arm_tests.cpp
		test/apraegldisplay_tests.cpp
		test/frame_factory_test_dma.cpp
		test/h264decoder_tests.cpp
	)
ENDIF(ENABLE_ARM64)

IF (ENABLE_CUDA)
	SET(CUDA_UT_FILES
		test/cudamemcopy_tests.cpp		
		test/resizenppi_tests.cpp
		test/rotatenppi_tests.cpp
		test/ccnppi_tests.cpp
		test/memtypeconversion_tests.cpp
	)
	IF(NOT ENABLE_ARM64) # following tests need CUDA but can not run on ARM ?

		SET(CUDA_UT_FILES ${CUDA_UT_FILES}	
			test/jpegencodernvjpeg_tests.cpp
			test/jpegdecodernvjpeg_tests.cpp			
			test/resizenppi_jpegencodernvjpeg_tests.cpp
			test/nvjpeg_combo_tests.cpp			
			test/overlaynppi_tests.cpp
			test/effectsnppi_tests.cpp
			test/h264Encodernvcodec_tests.cpp
			test/nv_mp4_file_tests.cpp
			test/nv_test_utils.h
			test/h264decoder_tests.cpp
		)
	ENDIF(NOT ENABLE_ARM64)
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
#	test/logger_stress_tests.cpp  #todo this test needs to be improved and added	
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
	test/faciallandmarkscv_tests.cpp
	test/imageviewermodule_tests.cpp
	test/ImageEncodeCV_tests.cpp
	test/rotatecv_tests.cpp
	test/affinetransform_tests.cpp
	test/brightness_contrast_tests.cpp
	test/virtualptz_tests.cpp
	test/webcam_source_tests.cpp
	test/facedetectorXform_tests.cpp
	test/sound_record_tests.cpp
	test/pullstratergy_tests.cpp
	test/QRReader_tests.cpp
	test/textoverlayxform_tests.cpp
	test/mp4writersink_tests.cpp
	test/pipeline_tests.cpp
#	test/multiple_pipeline_tests.cpp #todo this test needs to be improved and added
	test/valveModule_tests.cpp
	test/color_conversion_tests.cpp
	test/archivespacemanager_tests.cpp
	test/multimediaqueuexform_tests.cpp
  test/mp4readersource_tests.cpp
	test/rtsp_client_tests.cpp
	test/motionvector_extractor_and_overlay_tests.cpp
	test/mp4_reverse_play_tests.cpp
	test/ordered_cache_of_files_tests.cpp
	test/mp4_seek_tests.cpp
	test/mp4_simul_read_write_tests.cpp
	test/mp4_getlivevideots_tests.cpp
	test/mp4_dts_strategy_tests.cpp
	test/overlaymodule_tests.cpp
	test/testSignalGeneratorSrc_tests.cpp
	test/audioToTextXform_tests.cpp
	test/simpleControlModuleTests.cpp
	test/declarative/metadata_tests.cpp
	test/declarative/module_registry_tests.cpp
	test/declarative/pipeline_description_tests.cpp
	test/declarative/json_parser_tests.cpp
	test/declarative/module_factory_tests.cpp
	test/declarative/frame_type_registry_tests.cpp
	test/declarative/pipeline_validator_tests.cpp
	test/declarative/pipeline_analyzer_tests.cpp
	test/declarative/property_macros_tests.cpp
	test/declarative/module_registration_tests.cpp
	test/declarative/property_validators_tests.cpp
	test/declarative/pipeline_integration_tests.cpp
	${ARM64_UT_FILES}
	${CUDA_UT_FILES}
)

IF(ENABLE_LINUX)
	list(APPEND UT_FILES
	test/virtualcamerasink_tests.cpp
	test/QRReader_tests.cpp
	)
	# GTK-dependent tests only when GTK is available
	IF(GTK3_FOUND)
		list(APPEND UT_FILES test/gtkglrenderer_tests.cpp)
	ENDIF()
	set(GLEW_LIBRARIES
		GLEW::GLEW
		glfw
	)
ENDIF(ENABLE_LINUX)

add_executable(aprapipesut ${UT_FILES})

IF(ENABLE_ARM64)
	target_include_directories ( aprapipesut PRIVATE ${JETSON_MULTIMEDIA_LIB_INCLUDE} ${FFMPEG_ROOT} ${JPEG_INCLUDE_DIR})
ENDIF(ENABLE_ARM64)

IF (ENABLE_CUDA)
	target_include_directories ( aprapipesut PRIVATE ${NVCODEC_INCLUDE_DIR})
ENDIF (ENABLE_CUDA)

find_library(OPENH264_LIB NAMES openh264.lib libopenh264.a REQUIRED)
find_library(LIBMP4_LIB NAMES mp4lib.lib libmp4lib.a REQUIRED)

# ============================================================
# Common libraries shared by aprapipesut, aprapipes_cli, and apra_schema_generator
# DRY: Define once, use everywhere
# ============================================================
set(APRA_COMMON_LIBS
    ${GLEW_LIBRARIES}
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
    ${CMAKE_DL_LIBS}
    ${LIBARCHIVE_LIB}
    ZXing::Core
    ZXing::ZXing
    BZip2::BZip2
    ZLIB::ZLIB
    LibLZMA::LibLZMA
    bigint::bigint
    SFML::Audio
    whisper::whisper
)

IF(ENABLE_ARM64)
	target_include_directories(aprapipesut PRIVATE 	${VCPKG_GTK_INCLUDE_DIRS})
	# ARM64/Jetson needs GTK3 for GtkGlRenderer
	target_link_libraries(aprapipesut ${GTK3_LIBRARIES})
ENDIF(ENABLE_ARM64)

IF(ENABLE_LINUX AND NOT ENABLE_ARM64 AND GTK3_FOUND)
	# x64 Linux uses vcpkg's GTK
	target_include_directories(aprapipesut PRIVATE 	${GTK3_INCLUDE_DIRS})
	target_link_libraries(aprapipesut
	${GDK3_LIBRARIES}
  	${GTK3_LIBRARIES}
	)
ENDIF()

target_link_libraries(aprapipesut
  aprapipes
  ${APRA_COMMON_LIBS}
)

# Windows: Use /DELAYLOAD for CUDA DLLs so executable can start without CUDA installed
# This enables runtime detection of GPU availability - the DLLs are only loaded when first used
# Without this, the exe crashes immediately on systems without CUDA DLLs
IF(ENABLE_WINDOWS AND ENABLE_CUDA)
	# Link the delay load helper library
	target_link_libraries(aprapipesut delayimp.lib)

	# DELAYLOAD all CUDA DLLs that are linked via import libraries
	# The DLL names include version suffix (e.g., nvjpeg64_11.dll for CUDA 11.x)
	# Note: nvcuda.dll is NOT delay-loaded - it's loaded via LoadLibrary in CudaDriverLoader
	target_link_options(aprapipesut PRIVATE
		"/DELAYLOAD:nvjpeg64_11.dll"
		"/DELAYLOAD:nppig64_11.dll"
		"/DELAYLOAD:nppicc64_11.dll"
		"/DELAYLOAD:nppidei64_11.dll"
		"/DELAYLOAD:nppial64_11.dll"
		"/DELAYLOAD:nppc64_11.dll"
		"/DELAYLOAD:cublas64_11.dll"
		"/DELAYLOAD:cublasLt64_11.dll"
		"/DELAYLOAD:cudart64_110.dll"
		# Video Codec SDK DLLs - these are part of NVIDIA driver, not CUDA toolkit
		"/DELAYLOAD:nvcuvid.dll"
		"/DELAYLOAD:nvEncodeAPI64.dll"
	)
ENDIF(ENABLE_WINDOWS AND ENABLE_CUDA)

IF(ENABLE_WINDOWS)
	file(COPY ${RUNTIME_DLLS} DESTINATION Debug/)
	file(COPY ${RUNTIME_DLLS} DESTINATION Release/)
	IF(GHA)
		file(COPY ${RUNTIME_DLLS} DESTINATION RelWithDebInfo/)
	ENDIF(GHA)
ENDIF(ENABLE_WINDOWS)

include(GNUInstallDirs)

# BUILD_INTERFACE specifies where to find includes during build time
# here we set the include directory to be our src include directory
# as well as CMAKE_CURRENT_BINARY_DIR, which is where the generated
# calc_exported.h file is located.
# the command must be included in double quotes so the two directories,
# separated by a ';' can both be used (cmake needs it to be a string)
target_include_directories(
    aprapipes
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/;${CMAKE_CURRENT_BINARY_DIR}/>"
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set_target_properties(aprapipes PROPERTIES DEBUG_POSTFIX "d")


# specify the target to install (calculator library defined above)
# set the export name <name>-config (does not need to match target name)
# also specify where the .dylib/.so/.dll+.lib file should be installed
install(
    TARGETS aprapipes
    EXPORT aprapipes-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    EXPORT aprapipes-config
    NAMESPACE aprapipes::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aprapipes)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aprapipes)

# ============================================================
# ApraPipes CLI Tool (Declarative Pipeline)
# ============================================================
add_executable(aprapipes_cli
    tools/aprapipes_cli.cpp
)

# Force load aprapipes to include static module registrations
if(APPLE)
    target_link_libraries(aprapipes_cli PRIVATE
        -Wl,-force_load,$<TARGET_FILE:aprapipes>
    )
elseif(UNIX)
    target_link_libraries(aprapipes_cli PRIVATE
        -Wl,--whole-archive aprapipes -Wl,--no-whole-archive
    )
else()
    target_link_libraries(aprapipes_cli PRIVATE
        aprapipes
    )
endif()

# Link all required libraries (uses APRA_COMMON_LIBS - DRY)
target_link_libraries(aprapipes_cli PRIVATE ${APRA_COMMON_LIBS})

# GTK linkage for Linux/ARM64 (same as aprapipesut)
IF(ENABLE_ARM64)
    target_include_directories(aprapipes_cli PRIVATE ${VCPKG_GTK_INCLUDE_DIRS})
    target_link_libraries(aprapipes_cli PRIVATE ${GTK3_LIBRARIES})
ENDIF(ENABLE_ARM64)

IF(ENABLE_LINUX AND NOT ENABLE_ARM64 AND GTK3_FOUND)
    target_include_directories(aprapipes_cli PRIVATE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(aprapipes_cli PRIVATE ${GDK3_LIBRARIES} ${GTK3_LIBRARIES})
ENDIF()

# Include directories for declarative headers and dependencies
target_include_directories(aprapipes_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Install CLI executable
install(TARGETS aprapipes_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ============================================================
# ApraPipes Schema Generator Tool (E2)
# Exports module/frame type metadata to JSON/Markdown
# ============================================================
add_executable(apra_schema_generator
    tools/schema_generator.cpp
)

# Force load aprapipes to include static module registrations
if(APPLE)
    target_link_libraries(apra_schema_generator PRIVATE
        -Wl,-force_load,$<TARGET_FILE:aprapipes>
    )
elseif(UNIX)
    target_link_libraries(apra_schema_generator PRIVATE
        -Wl,--whole-archive aprapipes -Wl,--no-whole-archive
    )
else()
    target_link_libraries(apra_schema_generator PRIVATE
        aprapipes
    )
endif()

# Link all required libraries (uses APRA_COMMON_LIBS - DRY) + nlohmann_json for JSON output
target_link_libraries(apra_schema_generator PRIVATE
    ${APRA_COMMON_LIBS}
    nlohmann_json::nlohmann_json
)

# GTK linkage for Linux/ARM64 (same as aprapipesut)
IF(ENABLE_ARM64)
    target_include_directories(apra_schema_generator PRIVATE ${VCPKG_GTK_INCLUDE_DIRS})
    target_link_libraries(apra_schema_generator PRIVATE ${GTK3_LIBRARIES})
ENDIF(ENABLE_ARM64)

IF(ENABLE_LINUX AND NOT ENABLE_ARM64 AND GTK3_FOUND)
    target_include_directories(apra_schema_generator PRIVATE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(apra_schema_generator PRIVATE ${GDK3_LIBRARIES} ${GTK3_LIBRARIES})
ENDIF()

target_include_directories(apra_schema_generator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Install schema generator executable
install(TARGETS apra_schema_generator
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ============================================================
# Generate schema files at build time
# ============================================================
set(SCHEMA_DIR ${CMAKE_BINARY_DIR}/schema)

add_custom_command(
    OUTPUT ${SCHEMA_DIR}/modules.json ${SCHEMA_DIR}/frame_types.json
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SCHEMA_DIR}
    COMMAND apra_schema_generator
        --modules-json ${SCHEMA_DIR}/modules.json
        --frame-types-json ${SCHEMA_DIR}/frame_types.json
        --modules-md ${SCHEMA_DIR}/MODULES.md
        --frame-types-md ${SCHEMA_DIR}/FRAME_TYPES.md
    DEPENDS apra_schema_generator
    COMMENT "Generating ApraPipes schema from C++ metadata"
    VERBATIM
)

add_custom_target(generate_schema ALL
    DEPENDS ${SCHEMA_DIR}/modules.json ${SCHEMA_DIR}/frame_types.json
)

# ============================================================
# D2 Phase 3: Module Scanner for Registration Detection
# Scans headers for Module subclasses and generates a list
# Used by unit tests to detect unregistered modules
# ============================================================
set(MODULE_SUBCLASSES_FILE ${CMAKE_BINARY_DIR}/generated/module_subclasses.inc)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)

set(SCAN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ScanModuleClasses.cmake)

add_custom_target(scan-modules
    COMMAND ${CMAKE_COMMAND}
        -DSCAN_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include
        -DOUTPUT_FILE=${MODULE_SUBCLASSES_FILE}
        -P ${SCAN_SCRIPT}
    COMMENT "Scanning for Module subclasses..."
    VERBATIM
)

# Run scan at configure time to ensure file exists for builds
if(EXISTS ${SCAN_SCRIPT})
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -DSCAN_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include
            -DOUTPUT_FILE=${MODULE_SUBCLASSES_FILE}
            -P ${SCAN_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Add generated directory to test target include paths
# so module_registration_tests.cpp can find module_subclasses.inc
target_include_directories(aprapipesut PRIVATE ${CMAKE_BINARY_DIR}/generated)

# ============================================================
# Node.js Native Addon (Optional)
# Build with: cmake -DBUILD_NODE_ADDON=ON ...
# Requires: node-addon-api and node-api-headers from vcpkg
#
# When USE_FFMPEG_ASM_OPTIMIZATIONS=ON:
#   - FFmpeg-dependent modules are excluded from the Node.js addon
#   - Excluded modules: RTSPClientSrc, RTSPPusher, MotionVectorExtractor, H264ParserUtils
#   - This is necessary because FFmpeg with assembly optimizations cannot be
#     statically linked into shared objects due to -fPIC relocation issues
# ============================================================
option(BUILD_NODE_ADDON "Build Node.js native addon (@apralabs/aprapipes)" OFF)

if(BUILD_NODE_ADDON)
    message(STATUS "Node.js addon build enabled")

    # Find node-addon-api from vcpkg
    find_package(unofficial-node-addon-api CONFIG QUIET)
    find_package(unofficial-node-api-headers CONFIG QUIET)

    if(unofficial-node-addon-api_FOUND AND unofficial-node-api-headers_FOUND)
        message(STATUS "Found node-addon-api and node-api-headers")

        # When ASM optimizations are enabled, we need a separate library without FFmpeg modules
        if(USE_FFMPEG_ASM_OPTIMIZATIONS)
            message(STATUS "USE_FFMPEG_ASM_OPTIMIZATIONS=ON: Building Node.js addon WITHOUT FFmpeg-dependent modules")
            message(STATUS "  Excluded: RTSPClientSrc, RTSPPusher, MotionVectorExtractor, H264ParserUtils")

            # Build source list without FFmpeg-dependent modules
            set(NODE_COMPAT_SOURCE
                ${CORE_FILES} ${CORE_FILES_H}
                ${GENERIC_FILES} ${GENERIC_FILES_H}
                ${IP_FILES} ${IP_FILES_H}
            )
            IF(ENABLE_CUDA)
                set(NODE_COMPAT_SOURCE ${NODE_COMPAT_SOURCE}
                    ${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
                    ${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
                )
            ENDIF(ENABLE_CUDA)
            # Note: GTKGL_FILES intentionally excluded from Node.js addon
            # GtkGlRenderer is a display module not needed for headless pipelines
            # Including it requires GTK preload which causes libjpeg symbol conflicts

            # Create Node-compatible library without FFmpeg modules
            add_library(aprapipes_node_compat STATIC ${NODE_COMPAT_SOURCE})

            # Same include directories as aprapipes
            target_include_directories(aprapipes_node_compat PRIVATE
                ${JETSON_MULTIMEDIA_LIB_INCLUDE}
                ${GTK3_INCLUDE_DIRS}
                ${VCPKG_GTK_INCLUDE_DIRS}
                ${FFMPEG_INCLUDE_DIRS}
                ${OpenCV_INCLUDE_DIRS}
                ${Boost_INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
                ${CMAKE_BINARY_DIR}/generated
                ${LIBMP4_INC_DIR}
            )

            # Add compile definition to indicate FFmpeg modules are excluded
            target_compile_definitions(aprapipes_node_compat PRIVATE APRAPIPES_NO_FFMPEG_MODULES)

            set(NODE_ADDON_APRAPIPES_LIB aprapipes_node_compat)
            # Don't link FFmpeg libraries when using node_compat
            set(NODE_ADDON_EXTRA_LIBS "")
        else()
            # On Linux, we need to exclude GTK modules from Node.js addon
            # GTK display modules (GtkGlRenderer) can't work in Node.js anyway
            # Including them requires GTK preload which causes libjpeg symbol conflicts
            if(ENABLE_LINUX)
                message(STATUS "Building Node.js addon without GTK display modules (Linux)")

                # Build source list without GTK modules
                set(NODE_HEADLESS_SOURCE
                    ${CORE_FILES} ${CORE_FILES_H}
                    ${GENERIC_FILES} ${GENERIC_FILES_H}
                    ${IP_FILES} ${IP_FILES_H}
                )
                IF(ENABLE_CUDA)
                    set(NODE_HEADLESS_SOURCE ${NODE_HEADLESS_SOURCE}
                        ${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
                        ${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
                    )
                ENDIF(ENABLE_CUDA)
                # Note: GTKGL_FILES intentionally excluded - display modules not needed in Node.js

                add_library(aprapipes_node_headless STATIC ${NODE_HEADLESS_SOURCE})
                target_include_directories(aprapipes_node_headless PRIVATE
                    ${JETSON_MULTIMEDIA_LIB_INCLUDE}
                    ${FFMPEG_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
                    ${CMAKE_BINARY_DIR}/generated
                    ${LIBMP4_INC_DIR}
                    ${NVCODEC_INCLUDE_DIR}
                )
                target_compile_definitions(aprapipes_node_headless PRIVATE APRAPIPES_NO_GTK_MODULES)

                set(NODE_ADDON_APRAPIPES_LIB aprapipes_node_headless)
                set(NODE_ADDON_EXTRA_LIBS ${FFMPEG_LIBRARIES})
            else()
                message(STATUS "USE_FFMPEG_ASM_OPTIMIZATIONS=OFF: Building Node.js addon with all modules")
                set(NODE_ADDON_APRAPIPES_LIB aprapipes)
                set(NODE_ADDON_EXTRA_LIBS ${FFMPEG_LIBRARIES})
            endif()
        endif()

        set(NODE_ADDON_NAME "aprapipes_node")

        add_library(${NODE_ADDON_NAME} SHARED
            bindings/node/addon.cpp
            bindings/node/pipeline_wrapper.cpp
            bindings/node/module_wrapper.cpp
            bindings/node/event_emitter.cpp
        )

        # Set output extension to .node
        set_target_properties(${NODE_ADDON_NAME} PROPERTIES
            OUTPUT_NAME "aprapipes"
            PREFIX ""
            SUFFIX ".node"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )

        # Link node-addon-api
        target_link_libraries(${NODE_ADDON_NAME} PRIVATE
            unofficial::node-addon-api::node-addon-api
        )

        # Force load aprapipes (or aprapipes_node_compat) to include static module registrations
        if(APPLE)
            target_link_libraries(${NODE_ADDON_NAME} PRIVATE
                -Wl,-force_load,$<TARGET_FILE:${NODE_ADDON_APRAPIPES_LIB}>
            )
            # macOS: Use undefined dynamic lookup for Node symbols
            set_target_properties(${NODE_ADDON_NAME} PROPERTIES
                LINK_FLAGS "-undefined dynamic_lookup"
            )
        elseif(UNIX)
            target_link_libraries(${NODE_ADDON_NAME} PRIVATE
                -Wl,--whole-archive ${NODE_ADDON_APRAPIPES_LIB} -Wl,--no-whole-archive
            )
        else()
            target_link_libraries(${NODE_ADDON_NAME} PRIVATE
                ${NODE_ADDON_APRAPIPES_LIB}
            )
        endif()

        # Link all required libraries (uses APRA_COMMON_LIBS - DRY)
        # When ASM is ON, exclude FFmpeg libraries from node addon
        if(USE_FFMPEG_ASM_OPTIMIZATIONS)
            # Create a copy of APRA_COMMON_LIBS without FFMPEG_LIBRARIES
            set(NODE_ADDON_LIBS ${APRA_COMMON_LIBS})
            list(REMOVE_ITEM NODE_ADDON_LIBS ${FFMPEG_LIBRARIES})
            target_link_libraries(${NODE_ADDON_NAME} PRIVATE ${NODE_ADDON_LIBS})
        else()
            target_link_libraries(${NODE_ADDON_NAME} PRIVATE ${APRA_COMMON_LIBS})
        endif()

        # Include directories (same as aprapipes target + bindings/node for wrapper headers)
        target_include_directories(${NODE_ADDON_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/declarative
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/node
            ${OpenCV_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
            ${FFMPEG_INCLUDE_DIRS}
            ${LIBMP4_INC_DIR}
        )

        # Copy to project root for npm usage
        add_custom_command(TARGET ${NODE_ADDON_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:${NODE_ADDON_NAME}>
                ${CMAKE_CURRENT_SOURCE_DIR}/../$<TARGET_FILE_NAME:${NODE_ADDON_NAME}>
            COMMENT "Copying Node addon to project root"
        )

        # Install target
        install(TARGETS ${NODE_ADDON_NAME}
            LIBRARY DESTINATION lib/node
            RUNTIME DESTINATION lib/node
        )

        message(STATUS "Node.js addon target: ${NODE_ADDON_NAME}")
        message(STATUS "Node.js addon library: ${NODE_ADDON_APRAPIPES_LIB}")
    else()
        message(WARNING "Node.js addon requested but node-addon-api not found in vcpkg. Run: vcpkg install node-addon-api node-api-headers")
    endif()
endif()
