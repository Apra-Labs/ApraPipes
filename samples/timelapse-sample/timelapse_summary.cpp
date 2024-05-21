#include "Mp4ReaderSource.h"
#include "MotionVectorExtractor.h"
#include "Mp4WriterSink.h"
#include "H264Metadata.h"
#include "PipeLine.h"
#include "OverlayModule.h"
#include "ImageViewerModule.h"
#include "Mp4VideoMetadata.h"
#include "H264EncoderNVCodec.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "ColorConversionXForm.h"
#include "FileWriterModule.h"
#include "timelapse_summary.h"
#include <boost/test/unit_test.hpp>
#include "ExternalSinkModule.h"

TimelapsePipeline::TimelapsePipeline() : pipeline("test") {}

cudastream_sp cudaStream_ =
    boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
std::string outFolderPath = "C:\\Workspace\\ApraPipesFork\\ApraPipes\\data\\Mp4_videos\\timelapse_videos";
auto cuContext = apracucontext_sp(new ApraCUcontext());
uint32_t gopLength = 25;
uint32_t bitRateKbps = 1000;
uint32_t frameRate = 30;
H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
bool enableBFrames = false;
bool sendDecodedFrames = true;
bool sendOverlayFrames = false;
std::string videoPath = "C:\\Workspace\\ApraPipesFork\\ApraPipes\\data\\Mp4_videos\\h264_video_metadata\\20230514\\0011\\1707478361303.mp4";
auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
auto motionExtractorProps = MotionVectorExtractorProps(MotionVectorExtractorProps::MVExtractMethod::OPENH264, sendDecodedFrames, 10, sendOverlayFrames);
auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(motionExtractorProps));
auto colorchange1 = boost::shared_ptr<ColorConversion>(new ColorConversion(
    ColorConversionProps(ColorConversionProps::BGR_TO_RGB)));
auto colorchange2 = boost::shared_ptr<ColorConversion>(new ColorConversion(
    ColorConversionProps(ColorConversionProps::RGB_TO_YUV420PLANAR)));
auto sync = boost::shared_ptr<Module>(
    new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
auto encoder =
    boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(
        bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 10, 24, outFolderPath, true);
auto mp4WriterSink =
    boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));

void TimelapsePipeline::test() {
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    BOOST_TEST(mp4Reader->init());
    BOOST_TEST(motionExtractor->init());
    BOOST_TEST(colorchange1->init());
    BOOST_TEST(colorchange2->init());
    BOOST_TEST(sync->init());
    BOOST_TEST(encoder->init());
    BOOST_TEST(mp4WriterSink->init());
    mp4Reader->step();
    motionExtractor->step();
    colorchange1->step();
    colorchange2->step();
    sync->step();
    encoder->step();
    encoder->setNext(sink);
    auto frames = sink->pop();
    BOOST_CHECK_EQUAL(frames.size(), 1);
}

bool TimelapsePipeline::setupPipeline() {
    mp4Reader -> addOutPutPin(h264ImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
    mp4Reader->addOutPutPin(mp4Metadata);
    std::vector<std::string> mImagePin;
    mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
    mp4Reader->setNext(motionExtractor, mImagePin);
    motionExtractor -> setNext(colorchange1);
    colorchange1 -> setNext(colorchange2);
    auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_)));
    colorchange2 -> setNext(copy);
    copy -> setNext(sync);
    sync -> setNext(encoder);
    encoder -> setNext(mp4WriterSink);

    pipeline.appendModule(mp4Reader);
    pipeline.init();
    return true;
}

bool TimelapsePipeline::startPipeline() {
    pipeline.run_all_threaded();
    return true;
}

bool TimelapsePipeline::stopPipeline() {
    pipeline.stop();
    pipeline.term();
    pipeline.wait_for_all();
    return true;
}

int main() {
  LoggerProps loggerProps;
  loggerProps.logLevel = boost::log::trivial::severity_level::info;
  Logger::setLogLevel(boost::log::trivial::severity_level::info);
  Logger::initLogger(loggerProps);

  TimelapsePipeline pipelineInstance;

  // Setup the pipeline
  if (!pipelineInstance.setupPipeline()) {
    std::cerr << "Failed to setup pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  // Start the pipeline
  if (!pipelineInstance.startPipeline()) {
    std::cerr << "Failed to start pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(10));

  // Stop the pipeline
  if (!pipelineInstance.stopPipeline()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  return 0; // Or any success code
}
