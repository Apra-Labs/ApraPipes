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

TimelapsePipeline::TimelapsePipeline()
    : pipeline("test"), cudaStream_(new ApraCudaStream()),
      cuContext(new ApraCUcontext()),
      h264ImageMetadata(new H264Metadata(0, 0)) {}

bool TimelapsePipeline::setupPipeline(const std::string &videoPath,
                                      const std::string &outFolderPath) {
  uint32_t gopLength = 25;
  uint32_t bitRateKbps = 1000;
  uint32_t frameRate = 30;
  H264EncoderNVCodecProps::H264CodecProfile profile =
      H264EncoderNVCodecProps::MAIN;
  bool enableBFrames = false;
  bool sendDecodedFrames = true;
  bool sendOverlayFrames = false;

  auto mp4ReaderProps =
      Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
  mp4Reader =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  auto motionExtractorProps = MotionVectorExtractorProps(
      MotionVectorExtractorProps::MVExtractMethod::OPENH264, sendDecodedFrames,
      10, sendOverlayFrames);
  motionExtractor = boost::shared_ptr<MotionVectorExtractor>(
      new MotionVectorExtractor(motionExtractorProps));
  colorchange1 = boost::shared_ptr<ColorConversion>(new ColorConversion(
      ColorConversionProps(ColorConversionProps::BGR_TO_RGB)));
  colorchange2 = boost::shared_ptr<ColorConversion>(new ColorConversion(
      ColorConversionProps(ColorConversionProps::RGB_TO_YUV420PLANAR)));
  sync = boost::shared_ptr<Module>(
      new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
  encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(
      H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate,
                              profile, enableBFrames)));
  auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 10, 24, outFolderPath, true);
  mp4WriterSink =
      boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));

  mp4Reader->addOutPutPin(h264ImageMetadata);
  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mp4Reader->addOutPutPin(mp4Metadata);
  std::vector<std::string> mImagePin =
      mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
  mp4Reader->setNext(motionExtractor, mImagePin);
  motionExtractor->setNext(colorchange1);
  colorchange1->setNext(colorchange2);
  auto copy = boost::shared_ptr<Module>(
      new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_)));
  colorchange2->setNext(copy);
  copy->setNext(sync);
  sync->setNext(encoder);
  encoder->setNext(mp4WriterSink);

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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <videoPath> <outFolderPath>" << std::endl;
    return 1;
  }

  std::string videoPath = argv[argc - 2];
  std::string outFolderPath = argv[argc - 1];

  LoggerProps loggerProps;
  loggerProps.logLevel = boost::log::trivial::severity_level::info;
  Logger::setLogLevel(boost::log::trivial::severity_level::info);
  Logger::initLogger(loggerProps);

  TimelapsePipeline pipelineInstance;

  // Setup the pipeline
  if (!pipelineInstance.setupPipeline(videoPath, outFolderPath)) {
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