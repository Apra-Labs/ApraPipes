#pragma once
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "ImageViewerModule.h"
#include "WebCamSource.h"
#include "FileWriterModule.h"
#include "H264EncoderNVCodec.h"
#include "CudaCommon.h"
#include "Module.h"
#include "PipeLine.h"
#include "H264EncoderNVCodecHelper.h"
#include "Mp4WriterSink.h"

class RecordWebcam {
public:
    RecordWebcam();

    // Setup and control functions
    bool setUpPipeLine(const int& cameraId, const std::string& videoPath);
    bool startPipeline();
    bool stopPipeline();

private:
    // Core pipeline
    PipeLine pipeLine;

    // Webcam Source
    boost::shared_ptr<WebCamSource> mSource;

    // Viewer path (RGB -> BGR -> ImageViewer)
    boost::shared_ptr<ColorConversion> mColorConversion;
    boost::shared_ptr<ImageViewerModule> mImageViewerSink;

    // Encoder path (RGB -> YUV420 -> Cuda -> Encode -> File)
    boost::shared_ptr<ColorConversion> mRGB2YUV;                // ColorConversion (RGB to YUV420)
    boost::shared_ptr<CudaMemCopy> mCudaMemCopy;            // CudaMemCopy (Host to Device)
    boost::shared_ptr<H264EncoderNVCodec> mH264Encoder;            // H264EncoderNVCodec
         // FileWriterModule
    boost::shared_ptr<ColorConversion> mBGRtoRGB;

    boost::shared_ptr<Module> mSync;
    boost::shared_ptr<H264EncoderNVCodec> mEncoder;
    boost::shared_ptr<Mp4WriterSink> mMp4WriterSink;
    boost::shared_ptr<CudaMemCopy> mCopy;
    boost::shared_ptr<CudaMemCopy> mCopy_gpu2cpu;

    

    // GPU-related
    cudastream_sp mCudaStream_;
    apracucontext_sp mCuContext;

    //cudastream_sp mCudaStream_1;
};
