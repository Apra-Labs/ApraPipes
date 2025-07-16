
#include "RecordWebcam.h"
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "ExternalSinkModule.h"
#include "FileWriterModule.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "FrameMetadata.h"
#include "H264Decoder.h"
#include "H264Metadata.h"
#include "ImageViewerModule.h"
#include "JPEGEncoderNVJPEG.h"
#include "WebCamSource.h"
#include "Mp4VideoMetadata.h"
#include "FramesMuxer.h"
#include "CudaCommon.h"
#include "Module.h"

#include <boost/filesystem.hpp>

#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "H264EncoderNVCodecHelper.h"
#include "Mp4WriterSink.h"



RecordWebcam::RecordWebcam()
    : pipeLine("WebcamPipeline") {}




bool RecordWebcam::setUpPipeLine(const int& cameraId, const std::string& videoPath) {
    std::cout << "[Pipeline] Setting up recording-only pipeline with camId = " << cameraId << std::endl;

    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    // --- [0] Initialize CUDA context and stream ---
    mCudaStream_ = cudastream_sp(new ApraCudaStream());
    mCuContext = apracucontext_sp(new ApraCUcontext());

    //mCudaStream_1 = cudastream_sp(new ApraCudaStream());

    if (!mCudaStream_ || !mCuContext) {
        std::cerr << "[ERROR] Failed to initialize CUDA stream or context!" << std::endl;
        return false;
    }

    //   [1] Webcam source (RGB output)  
    WebCamSourceProps webCamSourceprops(cameraId, 1280, 720, 30);
    webCamSourceprops.logHealth = true;
    webCamSourceprops.logHealthFrequency = 100;
    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));



    //  [2] Color conversion: RGB to YUV420 Planar  
    auto rgbToYuvProps = ColorConversionProps(ColorConversionProps::RGB_TO_YUV420PLANAR);
    rgbToYuvProps.logHealth = true;
    rgbToYuvProps.logHealthFrequency = 100;
    mRGB2YUV = boost::shared_ptr<ColorConversion>(new ColorConversion(rgbToYuvProps));
    mSource->setNext(mRGB2YUV);


    // [3] Host to Device copy  
    auto cudaProps = CudaMemCopyProps(cudaMemcpyHostToDevice, mCudaStream_);
    cudaProps.logHealth = true;
    cudaProps.logHealthFrequency = 100;
    cudaProps.sync = true;
    mCopy = boost::shared_ptr<CudaMemCopy>(
        new CudaMemCopy(cudaProps));
    mRGB2YUV->setNext(mCopy);

    //  [5] H264 Encoder (GPU)  
    uint32_t gopLength = 10;
    uint32_t bitRateKbps = 1000;
    uint32_t frameRate = 30;
     
    H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
    bool enableBFrames = false;
    auto h264encprops = H264EncoderNVCodecProps(mCuContext);
    h264encprops.logHealth = true;
    h264encprops.logHealthFrequency = 100;
    mEncoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(h264encprops));
    mCopy->setNext(mEncoder);
 
    // mSync = boost::shared_ptr<Module>(
    //     new CudaStreamSynchronize(CudaStreamSynchronizeProps(mCudaStream_)));
    // mCopy_gpu2cpu->setNext(mSync);

  

    // //  [6] Device to Host copy  
    // auto cuda_2Props = CudaMemCopyProps(cudaMemcpyDeviceToHost, mCudaStream_);
    // cuda_2Props.logHealth = true;
    // cuda_2Props.logHealthFrequency = 1;
    // mCopy_gpu2cpu = boost::shared_ptr<CudaMemCopy>(
    //     new CudaMemCopy(cuda_2Props));
    // mEncoder->setNext(mCopy_gpu2cpu);



    // mSync = boost::shared_ptr<Module>(
    //     new CudaStreamSynchronize(CudaStreamSynchronizeProps(mCudaStream_)));
    // mCopy_gpu2cpu->setNext(mSync);


    //  [7] MP4 File Writer Sink 
    std::string outFolderPath = "D:/New_ApraPipes_NEW/ApraPipes/samples/output/test.mp4"; // Pass mp4 file path D:\New_ApraPipes_NEW\ApraPipes\samples\output\test.mp4
    
    // auto mp4WriterSinkProps = Mp4WriterSinkProps();
    // mp4WriterSinkProps.baseFolder = outFolderPath;
    // mp4WriterSinkProps.logHealth = true;
    // mp4WriterSinkProps.logHealthFrequency = 1;

    auto mp4WriterSinkProps = Mp4WriterSinkProps(UINT32_MAX, 1, 24, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;

    mMp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
    //mCopy_gpu2cpu->setNext(mMp4WriterSink);
    mEncoder->setNext(mMp4WriterSink);

    return true;

}




bool RecordWebcam::startPipeline() {
    std::cout << "[Pipeline] Starting recording-only pipeline..." << std::endl;

    // Append modules to pipeline
    pipeLine.appendModule(mSource);
    pipeLine.init();

    pipeLine.run_all_threaded();

    std::cout << "[Pipeline] Recording-only pipeline started successfully.\n";
    return true;
}




bool RecordWebcam::stopPipeline() {
    pipeLine.stop();
    pipeLine.term();
    pipeLine.wait_for_all();
    return true;
}
