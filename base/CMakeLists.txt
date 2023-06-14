cmake_minimum_required(VERSION 3.22)

OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" OFF)
OPTION(ENABLE_WINDOWS "Use this switch to enable WINDOWS" OFF)
OPTION(ENABLE_GST "Use this switch to integrate Gstreamer" ON)

IF(ENABLE_ARM64) #no GST build for ARM64 yet
   set(ENABLE_GST OFF)
   set(ENABLE_GST OFF)
ENDIF(ENABLE_ARM64)

set(VCPKG_INSTALL_OPTIONS "--clean-after-build")
IF(ENABLE_CUDA)
   add_compile_definitions(APRA_CUDA_ENABLED)
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

#use /MP only for language CXX (and not CUDA) and MSVC for both targets

add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>)

set(CMAKE_CXX_STANDARD 14)

project(APRAPIPES)


set ( LIBMP4_LIB_DIR "../thirdparty/libmp4/build/")
set ( LIBMP4_INC_DIR "../thirdparty/libmp4/include/")

set ( LIBRE_LIB_DIR "../thirdparty/lib_re/build/")
set ( LIBRE_INC_DIR "../thirdparty/lib_re/include/")

set ( BARESIP_LIB_DIR "../thirdparty/baresip/build/")
set ( BARESIP_INC_DIR "../thirdparty/baresip/include/")
IF(ENABLE_LINUX AND ENABLE_GST)
   # no gst on arm64 yet, linux:x64 uses a non-vcpk build of gst
   set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/../thirdparty/gst-build/gst-build-1.16/outInstall/lib/x86_64-linux-gnu/pkgconfig/")
ENDIF(ENABLE_LINUX AND ENABLE_GST)

IF(ENABLE_WINDOWS)
   set (LIBMP4_LIB_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/libmp4/build/${CMAKE_BUILD_TYPE}")
   #set (LIBRE_LIB_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/lib_re/build/")
   #set (BARESIP_LIB_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/baresip/build/")
ENDIF(ENABLE_WINDOWS)

message(STATUS $ENV{PKG_CONFIG_PATH}">>>>>> PKG_CONFIG_PATH")

message(STATUS ${LIBMP4_LIB_DIR} ">>>>>> LIBMP4_LIB_DIR")
message(STATUS ${LIBMP4_INC_DIR} ">>>>>> LIBMP4_INC_DIR")
message(STATUS ${LIBRE_LIB_DIR} ">>>>>> LIBRE_LIB_DIR")
message(STATUS ${LIBRE_INC_DIR} ">>>>>> LIBRE_INC_DIR")
message(STATUS ${BARESIP_LIB_DIR} ">>>>>> BARESIP_LIB_DIR")
message(STATUS ${BARESIP_INC_DIR} ">>>>>> BARESIP_INC_DIR")

list(APPEND CMAKE_PREFIX_PATH ${LIBMP4_LIB_DIR})
find_library(LIBMP4_LIB NAMES mp4lib.lib libmp4lib.a REQUIRED)

list(APPEND CMAKE_PREFIX_PATH ${LIBRE_LIB_DIR})
find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)


list(APPEND CMAKE_PREFIX_PATH ${BARESIP_LIB_DIR})
find_library(BARESIP_LIB NAMES libbaresip.so REQUIRED)


find_package(PkgConfig REQUIRED)

IF(ENABLE_GST)
   pkg_check_modules(gstreamer-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-1.0) 
   pkg_check_modules(gobject-2.0 REQUIRED IMPORTED_TARGET GLOBAL gobject-2.0) 
   pkg_check_modules(glib-2.0 REQUIRED IMPORTED_TARGET GLOBAL glib-2.0) 
   pkg_check_modules(gio-2.0 REQUIRED IMPORTED_TARGET GLOBAL gio-2.0)
   pkg_check_modules(gstreamer-rtp-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-rtp-1.0) 
   pkg_check_modules(gstreamer-rtsp-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-rtsp-1.0) 
   pkg_check_modules(gstreamer-rtsp-server-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-rtsp-server-1.0) 
   pkg_check_modules(gstreamer-app-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-app-1.0)
   pkg_check_modules(gstreamer-net-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-net-1.0)
   pkg_check_modules(gstreamer-webrtc-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-webrtc-1.0)
   pkg_check_modules(libpcre REQUIRED IMPORTED_TARGET GLOBAL libpcre)
   pkg_check_modules(gthread-2.0 REQUIRED IMPORTED_TARGET GLOBAL gthread-2.0)
   pkg_check_modules(gmodule-2.0 REQUIRED IMPORTED_TARGET GLOBAL gmodule-2.0)
   pkg_check_modules(gstreamer-base-1.0 REQUIRED IMPORTED_TARGET GLOBAL gstreamer-base-1.0)
   IF(NOT ENABLE_WINDOWS)
       pkg_check_modules(json-glib-1.0 REQUIRED IMPORTED_TARGET GLOBAL json-glib-1.0)
       pkg_check_modules(libsoup-2.4 REQUIRED IMPORTED_TARGET GLOBAL libsoup-2.4)
   ENDIF(NOT ENABLE_WINDOWS)
  
ENDIF(ENABLE_GST)
  

find_package(Boost COMPONENTS system thread filesystem serialization log chrono unit_test_framework REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(liblzma REQUIRED)
find_package(FFMPEG REQUIRED)
find_package(ZXing CONFIG REQUIRED)
find_package(bigint CONFIG REQUIRED)
find_package(SFML COMPONENTS system window audio graphics CONFIG REQUIRED)

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
       include_directories(AFTER SYSTEM /usr/local/cuda/include)
   ELSEIF(ENABLE_LINUX)
       find_library(LIBNVCUVID libnvcuvid.so PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64 NO_DEFAULT_PATH)
       find_library(LIBNVENCODE libnvidia-encode.so PATHS ../thirdparty/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64 NO_DEFAULT_PATH)
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
   src/Mp4WriterSink.cpp
   src/Mp4WriterSinkUtils.cpp
   src/MultimediaQueueXform.cpp
   src/Mp4ReaderSource.cpp
   src/Mp4ReaderSourceUtils.cpp
   src/RTSPClientSrc.cpp
   src/RTSPClientSrc.cpp
	src/MotionVectorExtractor.cpp
	src/OverlayMotionVectors.cpp
   src/BaresipWebRTC.cpp
   src/BaresipDemo.cpp
   src/BaresipVideoSink.cpp
   src/BaresipSession.cpp
   src/BaresipVideoAdapter.cpp
   src/WebRTCSink.cpp
)

SET(CORE_FILES_H
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
   include/AudioCaptureSrc.h
   include/QRReader.h
   include/Mp4WriterSink.h
   include/Mp4WriterSinkUtils.h
   include/EncodedImageMetadata.h
   include/PropsChangeMetadata.h
   include/ValveModule.h
   include/ArchiveSpaceManager.h
   include/MultimediaQueueXform.h
   include/RTSPClientSrc.h
   include/H264Metadata.h
   include/Mp4ReaderSource.h
   include/Mp4ReaderSourceUtils.h
   include/RTSPClientSrc.h
   include/H264Metadata.h
	include/MotionVectorExtractor.h
	include/OverlayMotionVectors.h
   include/BaresipWebRTC.h
   include/BaresipDemo.h
   include/BaresipVideoSink.h
   include/BaresipVideoAdapter.h
   include/WebRTCSink.h
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
)


IF(ENABLE_GST)
   IF(NOT ENABLE_WINDOWS)
       list(APPEND IP_FILES
           src/GstWebRTCSink.cpp
       )
   ENDIF(NOT ENABLE_WINDOWS)
   list(APPEND IP_FILES
       src/GstOnvifRtspSink.cpp
   )
ENDIF(ENABLE_GST)

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
   include/BrightnessContrastControlXform.h
   include/VirtualPTZ.h
   include/WebCamSource.h
   include/ApraFaceInfo.h
   include/FaceDetectsInfo.h
   include/FaceDetectorXform.h
   include/TextOverlayXForm.h
   include/ColorConversionXForm.h
)


IF(ENABLE_GST)
   IF(NOT ENABLE_WINDOWS)
       list(APPEND IP_FILES
           include/GstWebRTCSink.h
       )
   ENDIF(NOT ENABLE_WINDOWS)
   list(APPEND IP_FILES
       include/GstOnvifRtspSink.h
   )
ENDIF(ENABLE_GST)

SET(CUDA_CORE_FILES
   src/apra_cudamalloc_allocator.cu
   src/apra_cudamallochost_allocator.cu
   src/CudaMemCopy.cpp
   src/CudaStreamSynchronize.cpp
   src/CuCtxSynchronize.cpp
   src/CudaCommon.cpp

)

SET(CUDA_CORE_FILES_H
   include/CudaStreamSynchronize.h
   include/CudaMemCopy.h
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
   ${IP_FILES} ${IP_FILES_H}
)

IF(ENABLE_CUDA)
   set(SOURCE ${SOURCE}
       ${CUDA_CORE_FILES} ${CUDA_CORE_FILES_H}
       ${CUDA_IP_FILES} ${CUDA_IP_FILES_H}
   )
ENDIF(ENABLE_CUDA)

message(STATUS "-------------Printing Soure file list-----------------${SOURCE}")
add_library(aprapipes STATIC ${SOURCE})

message(STATUS "-------------Printing Include folder-----------------${gstreamer-1.0_INCLUDE_DIRS}")
target_include_directories ( aprapipes PRIVATE
${JETSON_MULTIMEDIA_LIB_INCLUDE}
${FFMPEG_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${LIBMP4_INC_DIR}
${LIBRE_INC_DIR}
${BARESIP_INC_DIR}
${NVCODEC_INCLUDE_DIR}
)

IF(ENABLE_GST)
   target_include_directories ( aprapipes PRIVATE
       ${gstreamer-app-1.0_INCLUDE_DIRS}
       ${gstreamer-base-1.0_INCLUDE_DIRS}
       ${gstreamer-1.0_INCLUDE_DIRS}
       ${gstreamer-rtsp-server-1.0_INCLUDE_DIRS}
       ${gstreamer-webrtc-1.0_INCLUDE_DIRS}
       ${libpcre_INCLUDE_DIRS}
       ${glib-2.0_INCLUDE_DIRS}
       ${gstreamer-rtsp-1.0_INCLUDE_DIRS}
       ${gstreamer-rtp-1.0_INCLUDE_DIRS}
       ${gstreamer-net-1.0_INCLUDE_DIRS}
       ${gio-2.0_INCLUDE_DIRS}
       ${gmodule-2.0_INCLUDE_DIRS}
       ${gthread-2.0_INCLUDE_DIRS}
       ${gobject-2.0_INCLUDE_DIRS}
   )
   IF(NOT ENABLE_WINDOWS)
       target_include_directories( aprapipes PRIVATE
           ${gstreamer-webrtc-1.0_INCLUDE_DIRS}
           ${json-glib-1.0_INCLUDE_DIRS}
           ${libsoup-2.4_INCLUDE_DIRS}
       )
   ENDIF(NOT ENABLE_WINDOWS)
ENDIF(ENABLE_GST)


# aprapipes Unit Tests

IF (ENABLE_ARM64)
   SET(ARM64_UT_FILES
       test/jpegencoderl4tm_tests.cpp
       test/jpegdecoderl4tm_tests.cpp
#		test/l4tm_dec_enc_1_tests.cpp  #todo this test needs to be improved to add to jetson suite
       test/opencvresize_tests.cpp
       test/h264encoderv4l2helper_tests.cpp
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
   )
   IF(NOT ENABLE_ARM64) # following tests need CUDA but can not run on ARM ?

       SET(CUDA_UT_FILES ${CUDA_UT_FILES} 
           test/jpegencodernvjpeg_tests.cpp
           test/jpegdecodernvjpeg_tests.cpp           
           test/resizenppi_jpegencodernvjpeg_tests.cpp
           test/nvjpeg_combo_tests.cpp        
           test/ccnppi_tests.cpp
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
   test/ImageEncodeCV_tests.cpp
   test/rotatecv_tests.cpp
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
   test/rtsp_client_tests.cpp
	test/motionvector_extractor_and_overlay_tests.cpp
   test/baresip_webrtc_tests.cpp
   test/baresipvideosink_tests.cpp
   ${ARM64_UT_FILES}
   ${CUDA_UT_FILES}
)

IF(ENABLE_LINUX)
   list(APPEND UT_FILES
   test/virtualcamerasink_tests.cpp
   test/QRReader_tests.cpp
   )
ENDIF(ENABLE_LINUX)

IF(ENABLE_GST)
   SET(GST_LIBS       
       PkgConfig::gstreamer-app-1.0
       PkgConfig::gstreamer-base-1.0
       PkgConfig::gstreamer-1.0
       PkgConfig::gstreamer-rtsp-server-1.0
       PkgConfig::libpcre
       PkgConfig::glib-2.0
       PkgConfig::gstreamer-rtsp-1.0
       PkgConfig::gstreamer-rtp-1.0
       PkgConfig::gstreamer-net-1.0   
   )
   list(APPEND UT_FILES
       test/gstrtsponvifsink_tests.cpp
   )

   IF(NOT ENABLE_WINDOWS)
       list(APPEND GST_LIBS
           PkgConfig::gstreamer-webrtc-1.0
           PkgConfig::json-glib-1.0
           PkgConfig::libsoup-2.4
       )
       list(APPEND UT_FILES
           test/gstwebrtcsink_tests.cpp
       )
   ENDIF(NOT ENABLE_WINDOWS)
ENDIF(ENABLE_GST)


add_executable(aprapipesut ${UT_FILES})

IF(ENABLE_ARM64)
   target_include_directories ( aprapipesut PRIVATE ${JETSON_MULTIMEDIA_LIB_INCLUDE} ${FFMPEG_ROOT} ${JPEG_INCLUDE_DIR})
ENDIF(ENABLE_ARM64)

IF (ENABLE_CUDA)
   target_include_directories ( aprapipesut PRIVATE ${NVCODEC_INCLUDE_DIR})
ENDIF (ENABLE_CUDA)


find_library(OPENH264_LIB NAMES openh264.lib libopenh264.a REQUIRED)

target_link_libraries(aprapipesut  
 aprapipes 
 ${JPEG_LIBRARIES}
 ${LIBMP4_LIB}
 ${OPENH264_LIB}
 ${LIBRE_LIB}
 ${BARESIP_LIB}
 ${Boost_LIBRARIES}
 ${FFMPEG_LIBRARIES}
 ${OpenCV_LIBRARIES}
 ${JETSON_LIBS}
 ${NVCUDAToolkit_LIBS}
 ${NVCODEC_LIB}
 ${NVJPEGLIB_L4T}
 ${CURSES_LIBRARIES}
 ${GST_LIBS}
 ZXing::Core
 ZXing::ZXing 
 BZip2::BZip2
 ZLIB::ZLIB
 liblzma::liblzma
 bigint::bigint
 sfml-audio
 )

IF(ENABLE_WINDOWS)
   #file(GLOB RUNTIME_DLLS ../thirdparty/gst-build/gst-build-1.16/outInstall/bin/*.dll)
   file(COPY ${RUNTIME_DLLS} DESTINATION Debug/)
   file(COPY ${RUNTIME_DLLS} DESTINATION Release/)
   IF(GHA)
       file(COPY ${RUNTIME_DLLS} DESTINATION RelWithDebInfo/)
   ENDIF(GHA)
ENDIF(ENABLE_WINDOWS)
