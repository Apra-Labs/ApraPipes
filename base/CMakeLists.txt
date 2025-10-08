cmake_minimum_required(VERSION 3.29)

OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" OFF)
OPTION(ENABLE_WINDOWS "Use this switch to enable WINDOWS" OFF)
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/custom-overlay")
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
	set(VCPKG_PLATFORM_TOOLSET "v142" CACHE STRING "v142" FORCE)
ENDIF(ENABLE_WINDOWS)

IF(ENABLE_ARM64)
	add_compile_definitions(ARM64)
	# set(VCPKG_OVERLAY_PORTS ../vcpkg/ports/cudnn)
	set(VCPKG_OVERLAY_TRIPLETS ../vcpkg/triplets/community/arm64-linux.cmake)
	set(ENV{VCPKG_FORCE_SYSTEM_BINARIES} 1)
	set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
ENDIF(ENABLE_ARM64)

# ============================================================================
# Component-Based Build System
# ============================================================================
# Define available components
set(APRAPIPES_ALL_COMPONENTS
	CORE
	VIDEO
	IMAGE_PROCESSING
	CUDA_COMPONENT
	ARM64_COMPONENT
	WEBCAM
	QR
	AUDIO
	FACE_DETECTION
	GTK_RENDERING
	THUMBNAIL
	IMAGE_VIEWER
)

# User can specify which components to build
# Default is ALL (maintains backward compatibility)
set(ENABLE_COMPONENTS "ALL" CACHE STRING "Semicolon-separated list of components to build (or ALL)")

# Parse and validate components
if(ENABLE_COMPONENTS STREQUAL "ALL")
	# Build all components - maintain backward compatibility
	# But filter out platform-specific components if platform isn't enabled
	set(APRAPIPES_ENABLED_COMPONENTS ${APRAPIPES_ALL_COMPONENTS})

	# Remove ARM64_COMPONENT if ARM64 is not enabled
	if(NOT ENABLE_ARM64)
		list(REMOVE_ITEM APRAPIPES_ENABLED_COMPONENTS ARM64_COMPONENT)
	endif()

	# Remove GTK_RENDERING if Linux is not enabled
	if(NOT ENABLE_LINUX)
		list(REMOVE_ITEM APRAPIPES_ENABLED_COMPONENTS GTK_RENDERING)
	endif()

	# Remove CUDA_COMPONENT if CUDA is not enabled
	if(NOT ENABLE_CUDA)
		list(REMOVE_ITEM APRAPIPES_ENABLED_COMPONENTS CUDA_COMPONENT)
	endif()

	message(STATUS "Building ALL components (backward compatibility mode)")
else()
	# Split the user-provided component list
	string(REPLACE ";" " " COMPONENTS_STR "${ENABLE_COMPONENTS}")
	message(STATUS "Building selected components: ${COMPONENTS_STR}")
	set(APRAPIPES_ENABLED_COMPONENTS ${ENABLE_COMPONENTS})
endif()

# Initialize all component flags to OFF
foreach(component ${APRAPIPES_ALL_COMPONENTS})
	set(APRAPIPES_ENABLE_${component} OFF)
endforeach()

# Enable requested components
foreach(component ${APRAPIPES_ENABLED_COMPONENTS})
	# Validate component exists
	list(FIND APRAPIPES_ALL_COMPONENTS ${component} component_index)
	if(component_index EQUAL -1)
		message(FATAL_ERROR "Unknown component: ${component}. Valid components: ${APRAPIPES_ALL_COMPONENTS}")
	endif()
	set(APRAPIPES_ENABLE_${component} ON)
	message(STATUS "  - Enabling component: ${component}")
endforeach()

# Component dependency resolution
# CORE is always required
if(NOT APRAPIPES_ENABLE_CORE)
	message(FATAL_ERROR "CORE component is required but not enabled")
endif()

# VIDEO depends on CORE
if(APRAPIPES_ENABLE_VIDEO AND NOT APRAPIPES_ENABLE_CORE)
	message(FATAL_ERROR "VIDEO component requires CORE")
endif()

# IMAGE_PROCESSING depends on CORE
if(APRAPIPES_ENABLE_IMAGE_PROCESSING AND NOT APRAPIPES_ENABLE_CORE)
	message(FATAL_ERROR "IMAGE_PROCESSING component requires CORE")
endif()

# CUDA_COMPONENT depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_CUDA_COMPONENT)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "CUDA_COMPONENT requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "CUDA_COMPONENT requires IMAGE_PROCESSING")
	endif()
	if(NOT ENABLE_CUDA)
		message(FATAL_ERROR "CUDA_COMPONENT requires ENABLE_CUDA=ON")
	endif()
endif()

# ARM64_COMPONENT depends on CORE and CUDA_COMPONENT
if(APRAPIPES_ENABLE_ARM64_COMPONENT)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "ARM64_COMPONENT requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_CUDA_COMPONENT)
		message(FATAL_ERROR "ARM64_COMPONENT requires CUDA_COMPONENT")
	endif()
	if(NOT ENABLE_ARM64)
		message(FATAL_ERROR "ARM64_COMPONENT requires ENABLE_ARM64=ON")
	endif()
endif()

# WEBCAM depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_WEBCAM)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "WEBCAM component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "WEBCAM component requires IMAGE_PROCESSING")
	endif()
endif()

# QR depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_QR)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "QR component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "QR component requires IMAGE_PROCESSING")
	endif()
endif()

# AUDIO depends on CORE
if(APRAPIPES_ENABLE_AUDIO AND NOT APRAPIPES_ENABLE_CORE)
	message(FATAL_ERROR "AUDIO component requires CORE")
endif()

# FACE_DETECTION depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_FACE_DETECTION)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "FACE_DETECTION component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "FACE_DETECTION component requires IMAGE_PROCESSING")
	endif()
endif()

# GTK_RENDERING depends on CORE and IMAGE_PROCESSING (Linux only)
if(APRAPIPES_ENABLE_GTK_RENDERING)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "GTK_RENDERING component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "GTK_RENDERING component requires IMAGE_PROCESSING")
	endif()
	if(NOT ENABLE_LINUX)
		message(FATAL_ERROR "GTK_RENDERING component requires ENABLE_LINUX=ON")
	endif()
endif()

# THUMBNAIL depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_THUMBNAIL)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "THUMBNAIL component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "THUMBNAIL component requires IMAGE_PROCESSING")
	endif()
endif()

# IMAGE_VIEWER depends on CORE and IMAGE_PROCESSING
if(APRAPIPES_ENABLE_IMAGE_VIEWER)
	if(NOT APRAPIPES_ENABLE_CORE)
		message(FATAL_ERROR "IMAGE_VIEWER component requires CORE")
	endif()
	if(NOT APRAPIPES_ENABLE_IMAGE_PROCESSING)
		message(FATAL_ERROR "IMAGE_VIEWER component requires IMAGE_PROCESSING")
	endif()
endif()

# Add compile definitions for enabled components
foreach(component ${APRAPIPES_ENABLED_COMPONENTS})
	add_compile_definitions(APRAPIPES_ENABLE_${component})
endforeach()

message(STATUS "Component configuration complete")

# ============================================================================
# vcpkg Manifest Feature Mapping
# ============================================================================
# Map ENABLE_COMPONENTS to VCPKG_MANIFEST_FEATURES for conditional dependency installation
# This must be done before project() command since vcpkg runs during project initialization

# Create mapping from component names to vcpkg feature names
set(VCPKG_MANIFEST_FEATURES "")

if(ENABLE_COMPONENTS STREQUAL "ALL")
	# Enable all vcpkg features for backward compatibility
	list(APPEND VCPKG_MANIFEST_FEATURES "all")
	message(STATUS "vcpkg features: all (full backward-compatible build)")
else()
	# Map enabled components to vcpkg features
	foreach(component ${APRAPIPES_ENABLED_COMPONENTS})
		if(component STREQUAL "CORE")
			# CORE dependencies are always in base dependencies, no feature needed
		elseif(component STREQUAL "VIDEO")
			list(APPEND VCPKG_MANIFEST_FEATURES "video")
		elseif(component STREQUAL "IMAGE_PROCESSING")
			list(APPEND VCPKG_MANIFEST_FEATURES "image-processing")
		elseif(component STREQUAL "CUDA_COMPONENT")
			list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
		elseif(component STREQUAL "ARM64_COMPONENT")
			# ARM64 doesn't use vcpkg, uses system libraries
		elseif(component STREQUAL "WEBCAM")
			list(APPEND VCPKG_MANIFEST_FEATURES "webcam")
		elseif(component STREQUAL "QR")
			list(APPEND VCPKG_MANIFEST_FEATURES "qr")
		elseif(component STREQUAL "AUDIO")
			list(APPEND VCPKG_MANIFEST_FEATURES "audio")
		elseif(component STREQUAL "FACE_DETECTION")
			list(APPEND VCPKG_MANIFEST_FEATURES "face-detection")
		elseif(component STREQUAL "GTK_RENDERING")
			list(APPEND VCPKG_MANIFEST_FEATURES "gtk-rendering")
		elseif(component STREQUAL "THUMBNAIL")
			list(APPEND VCPKG_MANIFEST_FEATURES "thumbnail")
		elseif(component STREQUAL "IMAGE_VIEWER")
			list(APPEND VCPKG_MANIFEST_FEATURES "image-viewer")
		endif()
	endforeach()

	# Display enabled vcpkg features
	if(VCPKG_MANIFEST_FEATURES)
		string(REPLACE ";" ", " FEATURES_STR "${VCPKG_MANIFEST_FEATURES}")
		message(STATUS "vcpkg features: ${FEATURES_STR}")
	else()
		message(STATUS "vcpkg features: none (CORE-only build with base dependencies)")
	endif()
endif()

# Note: VCPKG_MANIFEST_FEATURES is automatically used by vcpkg during project() initialization
# ============================================================================

#use /MP only for language CXX (and not CUDA) and MSVC for both targets

add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>)

set(CMAKE_CXX_STANDARD 17)

IF(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
ENDIF()

project(APRAPIPES)

message(STATUS $ENV{PKG_CONFIG_PATH}">>>>>> PKG_CONFIG_PATH")

# ============================================================================
# Component-Based Dependency Resolution
# ============================================================================
find_package(PkgConfig REQUIRED)

# Add custom cmake modules directory for FindCUDA.cmake compatibility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- CORE Component Dependencies (Always required) ---
find_package(Boost COMPONENTS system thread filesystem serialization log chrono unit_test_framework REQUIRED)
find_package(JPEG REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(bigint CONFIG REQUIRED)

# --- IMAGE_PROCESSING Component Dependencies ---
if(APRAPIPES_ENABLE_IMAGE_PROCESSING OR APRAPIPES_ENABLE_CUDA_COMPONENT OR APRAPIPES_ENABLE_ARM64_COMPONENT)
	find_package(OpenCV CONFIG REQUIRED)
endif()

# --- VIDEO Component Dependencies ---
if(APRAPIPES_ENABLE_VIDEO)
	find_package(FFMPEG REQUIRED)
endif()

# --- QR Component Dependencies ---
if(APRAPIPES_ENABLE_QR)
	find_package(ZXing CONFIG REQUIRED)
endif()

# --- AUDIO Component Dependencies ---
if(APRAPIPES_ENABLE_AUDIO)
	find_package(SFML COMPONENTS system window audio graphics CONFIG REQUIRED)
	find_package(whisper CONFIG REQUIRED)
endif()

# --- GTK_RENDERING Component Dependencies (Linux only) ---
if(APRAPIPES_ENABLE_GTK_RENDERING AND ENABLE_LINUX)
	find_package(GLEW REQUIRED)
	find_package(glfw3 CONFIG REQUIRED)
	find_package(FreeGLUT CONFIG REQUIRED)
	pkg_check_modules(GIO REQUIRED gio-2.0)
	pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
	pkg_check_modules(GLFW REQUIRED glfw3)
	pkg_check_modules(GDK3 REQUIRED gdk-3.0)
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
endif()

# Platform-specific package config paths
IF(ENABLE_ARM64)
	set(ENV{PKG_CONFIG_PATH} "/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig")
ENDIF(ENABLE_ARM64)

message(STATUS "Dependency resolution complete")
# ============================================================================

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
		find_library(NVBUFUTILSLIB nvbuf_utils REQUIRED)
		find_library(EGLSTREAM_CAMCONSUMER_LIB nveglstream_camconsumer REQUIRED)
		find_library(NVARGUS_SOCKETCLINET_LIB nvargus_socketclient REQUIRED)
		find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)
		find_library(BARESIP_LIB NAMES libbaresip.so REQUIRED)
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
			${EGLSTREAM_CAMCONSUMER_LIB}
			${NVARGUS_SOCKETCLINET_LIB}
		)
		include_directories(AFTER SYSTEM /usr/local/cuda/include)
		include_directories(AFTER SYSTEM  /usr/local/cuda/samples/common/inc/)
		include_directories(AFTER SYSTEM  /usr/include/)
	    include_directories(AFTER SYSTEM  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ApraGTKUtils/includes/)
	ELSEIF(ENABLE_LINUX)
		find_library(LIBNVCUVID libnvcuvid.so PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
		find_library(LIBNVENCODE libnvidia-encode.so PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
		find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)
		find_library(BARESIP_LIB NAMES libbaresip.so REQUIRED)
		SET(NVCODEC_LIB ${LIBNVCUVID} ${LIBNVENCODE})
		SET(NVCUDAToolkit_LIBS
			libcuda.so		
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

# ApraPipes library - Component-Based Source Organization
# ============================================================================

# --- CORE Component (Always built) ---
SET(COMPONENT_CORE_FILES
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
	src/SimpleControlModule.cpp
	src/APErrorObject.cpp
	src/APHealthObject.cpp
	src/AbsControlModule.cpp
	src/ValveModule.cpp
	src/TestSignalGeneratorSrc.cpp
)

# --- VIDEO Component ---
SET(COMPONENT_VIDEO_FILES
	src/Mp4WriterSink.cpp
	src/Mp4WriterSinkUtils.cpp
	src/Mp4ReaderSource.cpp
	src/MultimediaQueueXform.cpp
	src/RTSPClientSrc.cpp
	src/RTSPPusher.cpp
	src/H264FrameDemuxer.cpp
	src/H264ParserUtils.cpp
	src/H264Utils.cpp
	src/MotionVectorExtractor.cpp
	src/OrderedCacheOfFiles.cpp
)

# --- IMAGE_PROCESSING Component ---
SET(COMPONENT_IMAGE_PROCESSING_FILES
	src/ApraLines.cpp
	src/CalcHistogramCV.cpp
	src/HistogramOverlay.cpp
	src/ImageDecoderCV.cpp
	src/BMPConverter.cpp
	src/ImageResizeCV.cpp
	src/ImageEncoderCV.cpp
	src/RotateCV.cpp
	src/AffineTransform.cpp
	src/BrightnessContrastControlXform.cpp
	src/VirtualPTZ.cpp
	src/ColorConversionXForm.cpp
	src/AbsColorConversionFactory.cpp
	src/ColorConversionStrategy.h
	src/AbsColorConversionFactory.h
	src/ArchiveSpaceManager.cpp
	src/Overlay.cpp
	src/OverlayFactory.h
	src/OverlayFactory.cpp
	src/OverlayModule.cpp
	src/TextOverlayXForm.cpp
)

# --- AUDIO Component ---
SET(COMPONENT_AUDIO_FILES
	src/AudioCaptureSrc.cpp
	src/AudioToTextXForm.cpp
)

# --- QR Component ---
SET(COMPONENT_QR_FILES
	src/QRReader.cpp
)

# --- WEBCAM Component ---
SET(COMPONENT_WEBCAM_FILES
	src/WebCamSource.cpp
)

# --- FACE_DETECTION Component ---
SET(COMPONENT_FACE_DETECTION_FILES
	src/FaceDetectorXform.cpp
	src/FacialLandmarksCV.cpp
)

# --- THUMBNAIL Component ---
SET(COMPONENT_THUMBNAIL_FILES
	src/ThumbnailListGenerator.cpp
)

# --- IMAGE_VIEWER Component ---
SET(COMPONENT_IMAGE_VIEWER_FILES
	src/ImageViewerModule.cpp
)

# --- CORE Component Headers ---
SET(COMPONENT_CORE_FILES_H
	include/BufferMaker.h
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
	include/EncodedImageMetadata.h
	include/PropsChangeMetadata.h
	include/ValveModule.h
	include/TestSignalGeneratorSrc.h
	include/AbsControlModule.h
	include/SimpleControlModule.h
	include/APErrorObject.h
	include/APCallback.h
	include/APHealthObject.h
)

IF(ENABLE_WINDOWS)
	list(APPEND COMPONENT_CORE_FILES_H
		include/targetver.h
		include/stdafx.h
	)
ENDIF(ENABLE_WINDOWS)

# --- VIDEO Component Headers ---
SET(COMPONENT_VIDEO_FILES_H
	include/Mp4ErrorFrame.h
	include/Mp4WriterSink.h
	include/Mp4WriterSinkUtils.h
	include/Mp4ReaderSource.h
	include/MultimediaQueueXform.h
	include/RTSPClientSrc.h
	include/RTSPPusher.h
	include/H264FrameDemuxer.h
	include/H264ParserUtils.h
	include/H264Utils.h
	include/H264Metadata.h
	include/MotionVectorExtractor.h
	include/OrderedCacheOfFiles.h
)

# --- IMAGE_PROCESSING Component Headers ---
SET(COMPONENT_IMAGE_PROCESSING_FILES_H
	include/HistogramOverlay.h
	include/CalcHistogramCV.h
	include/ApraPoint2f.h
	include/ApraLines.h
	include/ImageDecoderCV.h
	include/BMPConverter.h
	include/ImageResizeCV.h
	include/ImageEncoderCV.h
	include/RotateCV.h
	include/AffineTransform.h
	include/BrightnessContrastControlXform.h
	include/VirtualPTZ.h
	include/ColorConversionXForm.h
	include/Overlay.h
	include/OverlayModule.h
	include/TextOverlayXForm.h
	include/ArchiveSpaceManager.h
)

# --- AUDIO Component Headers ---
SET(COMPONENT_AUDIO_FILES_H
	include/AudioCaptureSrc.h
	include/AudioToTextXForm.h
)

# --- QR Component Headers ---
SET(COMPONENT_QR_FILES_H
	include/QRReader.h
)

# --- WEBCAM Component Headers ---
SET(COMPONENT_WEBCAM_FILES_H
	include/WebCamSource.h
)

# --- FACE_DETECTION Component Headers ---
SET(COMPONENT_FACE_DETECTION_FILES_H
	include/FaceDetectorXform.h
	include/FacialLandmarksCV.h
	include/ApraFaceInfo.h
	include/FaceDetectsInfo.h
)

# --- THUMBNAIL Component Headers ---
SET(COMPONENT_THUMBNAIL_FILES_H
	include/ThumbnailListGenerator.h
)

# --- IMAGE_VIEWER Component Headers ---
SET(COMPONENT_IMAGE_VIEWER_FILES_H
	include/ImageViewerModule.h
)

# --- GTK_RENDERING Component (Linux only) ---
SET(COMPONENT_GTK_RENDERING_FILES
	src/Background.cpp
	src/GtkGlRenderer.cpp
	src/GTKMatrix.cpp
	src/GTKModel.cpp
	src/GTKSetup.cpp
	src/GTKView.cpp
	src/VirtualCameraSink.cpp
)

SET(COMPONENT_GTK_RENDERING_FILES_H
	include/Background.h
	include/GLUtils.h
	include/GtkGlRenderer.h
	include/GTKMatrix.h
	include/GTKModel.h
	include/GTKSetup.h
	include/GTKView.h
	include/VirtualCameraSink.h
)

# Linux-specific additions
IF(ENABLE_LINUX)
	list(APPEND COMPONENT_CORE_FILES src/KeyboardListener.cpp)
	list(APPEND COMPONENT_CORE_FILES_H include/KeyboardListener.h)
ENDIF(ENABLE_LINUX)

# --- CUDA_COMPONENT (GPU acceleration) ---
SET(COMPONENT_CUDA_FILES
	src/apra_cudamalloc_allocator.cu
	src/apra_cudamallochost_allocator.cu
	src/CudaMemCopy.cpp
	src/MemTypeConversion.cpp
	src/CudaStreamSynchronize.cpp
	src/CuCtxSynchronize.cpp
	src/CudaCommon.cpp
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

SET(COMPONENT_CUDA_FILES_H
	include/CudaStreamSynchronize.h
	include/CudaMemCopy.h
	include/MemTypeConversion.h
	include/apra_cudamallochost_allocator.h
	include/apra_cudamalloc_allocator.h
	include/CuCtxSynchronize.h
	include/CudaCommon.h
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

# Add platform-specific CUDA modules (non-ARM64: NVJPEG/NVCodec)
IF(NOT ENABLE_ARM64)
	list(APPEND COMPONENT_CUDA_FILES
		src/JPEGDecoderNVJPEG.cpp
		src/JPEGEncoderNVJPEG.cpp
		src/H264EncoderNVCodecHelper.cpp
		src/H264EncoderNVCodec.cpp
		src/H264DecoderNvCodecHelper.cpp
		src/H264DecoderNvCodecHelper.h
	)
	list(APPEND COMPONENT_CUDA_FILES_H
		include/JPEGEncoderNVJPEG.h
		include/JPEGDecoderNVJPEG.h
		include/H264EncoderNVCodecHelper.h
		include/H264EncoderNVCodec.h
	)
ENDIF(NOT ENABLE_ARM64)

# --- ARM64_COMPONENT (Jetson-specific) ---
SET(COMPONENT_ARM64_FILES
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

SET(COMPONENT_ARM64_FILES_H
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

# ============================================================================
# Build SOURCE list based on enabled components
# ============================================================================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Initialize with empty SOURCE
set(SOURCE "")

# CORE is always included
if(APRAPIPES_ENABLE_CORE)
	set(SOURCE ${SOURCE}
		${COMPONENT_CORE_FILES}
		${COMPONENT_CORE_FILES_H}
	)
endif()

# VIDEO component
if(APRAPIPES_ENABLE_VIDEO)
	set(SOURCE ${SOURCE}
		${COMPONENT_VIDEO_FILES}
		${COMPONENT_VIDEO_FILES_H}
	)
endif()

# IMAGE_PROCESSING component
if(APRAPIPES_ENABLE_IMAGE_PROCESSING)
	set(SOURCE ${SOURCE}
		${COMPONENT_IMAGE_PROCESSING_FILES}
		${COMPONENT_IMAGE_PROCESSING_FILES_H}
	)
endif()

# CUDA_COMPONENT
if(APRAPIPES_ENABLE_CUDA_COMPONENT)
	set(SOURCE ${SOURCE}
		${COMPONENT_CUDA_FILES}
		${COMPONENT_CUDA_FILES_H}
	)
endif()

# ARM64_COMPONENT
if(APRAPIPES_ENABLE_ARM64_COMPONENT)
	set(SOURCE ${SOURCE}
		${COMPONENT_ARM64_FILES}
		${COMPONENT_ARM64_FILES_H}
	)
endif()

# WEBCAM component
if(APRAPIPES_ENABLE_WEBCAM)
	set(SOURCE ${SOURCE}
		${COMPONENT_WEBCAM_FILES}
		${COMPONENT_WEBCAM_FILES_H}
	)
endif()

# QR component
if(APRAPIPES_ENABLE_QR)
	set(SOURCE ${SOURCE}
		${COMPONENT_QR_FILES}
		${COMPONENT_QR_FILES_H}
	)
endif()

# AUDIO component
if(APRAPIPES_ENABLE_AUDIO)
	set(SOURCE ${SOURCE}
		${COMPONENT_AUDIO_FILES}
		${COMPONENT_AUDIO_FILES_H}
	)
endif()

# FACE_DETECTION component
if(APRAPIPES_ENABLE_FACE_DETECTION)
	set(SOURCE ${SOURCE}
		${COMPONENT_FACE_DETECTION_FILES}
		${COMPONENT_FACE_DETECTION_FILES_H}
	)
endif()

# GTK_RENDERING component (Linux only)
if(APRAPIPES_ENABLE_GTK_RENDERING)
	set(SOURCE ${SOURCE}
		${COMPONENT_GTK_RENDERING_FILES}
		${COMPONENT_GTK_RENDERING_FILES_H}
	)
endif()

# THUMBNAIL component
if(APRAPIPES_ENABLE_THUMBNAIL)
	set(SOURCE ${SOURCE}
		${COMPONENT_THUMBNAIL_FILES}
		${COMPONENT_THUMBNAIL_FILES_H}
	)
endif()

# IMAGE_VIEWER component
if(APRAPIPES_ENABLE_IMAGE_VIEWER)
	set(SOURCE ${SOURCE}
		${COMPONENT_IMAGE_VIEWER_FILES}
		${COMPONENT_IMAGE_VIEWER_FILES_H}
	)
endif()

list(LENGTH SOURCE SOURCE_FILE_COUNT)
message(STATUS "Building with ${SOURCE_FILE_COUNT} source files")
# ============================================================================

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
	${ARM64_UT_FILES}
	${CUDA_UT_FILES}
)

IF(ENABLE_LINUX)
	list(APPEND UT_FILES
	test/gtkglrenderer_tests.cpp
	test/virtualcamerasink_tests.cpp
	test/QRReader_tests.cpp
	)
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

# ============================================================================
# Component-Based Linking for Test Executable
# ============================================================================

# Setup include directories based on components
IF(ENABLE_ARM64 AND APRAPIPES_ENABLE_ARM64_COMPONENT)
	target_include_directories(aprapipesut PRIVATE ${VCPKG_GTK_INCLUDE_DIRS})
ENDIF()

IF(ENABLE_LINUX AND APRAPIPES_ENABLE_GTK_RENDERING)
	target_include_directories(aprapipesut PRIVATE ${GTK3_INCLUDE_DIRS})
ENDIF()

# Start with core libraries (always linked)
target_link_libraries(aprapipesut
  aprapipes
  ${JPEG_LIBRARIES}
  ${Boost_LIBRARIES}
  BZip2::BZip2
  ZLIB::ZLIB
  LibLZMA::LibLZMA
  bigint::bigint
)

# VIDEO Component libraries
if(APRAPIPES_ENABLE_VIDEO)
	target_link_libraries(aprapipesut
		${LIBMP4_LIB}
		${OPENH264_LIB}
		${FFMPEG_LIBRARIES}
	)
endif()

# IMAGE_PROCESSING Component libraries
if(APRAPIPES_ENABLE_IMAGE_PROCESSING OR APRAPIPES_ENABLE_CUDA_COMPONENT OR APRAPIPES_ENABLE_ARM64_COMPONENT)
	target_link_libraries(aprapipesut
		${OpenCV_LIBRARIES}
	)
endif()

# CUDA_COMPONENT libraries
if(APRAPIPES_ENABLE_CUDA_COMPONENT)
	target_link_libraries(aprapipesut
		${NVCUDAToolkit_LIBS}
		${NVCODEC_LIB}
	)
endif()

# ARM64_COMPONENT libraries
if(APRAPIPES_ENABLE_ARM64_COMPONENT)
	target_link_libraries(aprapipesut
		${JETSON_LIBS}
		${NVJPEGLIB_L4T}
		${CURSES_LIBRARIES}
	)
endif()

# QR Component libraries
if(APRAPIPES_ENABLE_QR)
	target_link_libraries(aprapipesut
		ZXing::Core
		ZXing::ZXing
	)
endif()

# AUDIO Component libraries
if(APRAPIPES_ENABLE_AUDIO)
	target_link_libraries(aprapipesut
		sfml-audio
		whisper::whisper
	)
endif()

# GTK_RENDERING Component libraries
if(APRAPIPES_ENABLE_GTK_RENDERING AND ENABLE_LINUX)
	target_link_libraries(aprapipesut
		${GLEW_LIBRARIES}
		${GDK3_LIBRARIES}
		${GTK3_LIBRARIES}
	)
endif()

message(STATUS "Linking configuration complete")
# ============================================================================

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
