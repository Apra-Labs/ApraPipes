
#include "PlayMp4VideoFromBeginning.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "H264Decoder.h"
#include "ValveModule.h"
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "FileWriterModule.h"
#include "JPEGEncoderNVJPEG.h"
#include "ExternalSinkModule.h"
#include "ImageViewerModule.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "../../base/test/test_utils.h"

class SinkModuleProps : public ModuleProps {
public:
  SinkModuleProps() : ModuleProps(){};
};

class SinkModule : public Module {
public:
  SinkModule(SinkModuleProps props) : Module(SINK, "sinkModule", props){};
  boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
  frame_container pop() { return Module::pop(); }

protected:
  bool process() { return false; }
  bool validateOutputPins() { return true; }
  bool validateInputPins() { return true; }
};

PlayMp4VideoFromBeginning::PlayMp4VideoFromBeginning() : pipeLine("pipeline") {

}

bool PlayMp4VideoFromBeginning::testPipeLineForFlushQue() {
  auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
  colorchange->setNext(sink);

  BOOST_CHECK(mp4Reader->init());
  BOOST_CHECK(decoder->init());
  BOOST_CHECK(colorchange->init());
  BOOST_CHECK(sink->init());

  for (int i = 0; i <= 20; i++) {
    mp4Reader->step();
    decoder->step();
  }
  colorchange->step();

  auto frames = sink->pop();
  auto sinkQue = sink->getQue();
  BOOST_TEST(sinkQue->size() == 1);
  sinkQue->flush();
  BOOST_TEST(sinkQue->size() == 0);
  frame_sp outputFrame = frames.cbegin()->second;

 
    
  Test_Utils::saveOrCompare("../.././data/mp4Reader_saveOrCompare/h264/testplay.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
  return true;
}
bool PlayMp4VideoFromBeginning::testPipeLineForSeek() {
  auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
  mp4Reader->setNext(sink);

  BOOST_CHECK(mp4Reader->init());
  BOOST_CHECK(decoder->init());
  BOOST_CHECK(colorchange->init());
  BOOST_CHECK(sink->init());

  mp4Reader->step();

  auto frames = sink->pop();
  auto imgFrame = frames.begin()->second;

  uint64_t skipTS = 1686723797848;
  mp4Reader->randomSeek(skipTS, false);
  mp4Reader->step();
  BOOST_TEST(imgFrame->timestamp == 1686723797856);
  LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS
           << " msecs later from skipTS";

 return true;
}
bool PlayMp4VideoFromBeginning::setUpPipeLine() {
    // Implementation
    std::string videoPath = "../.././data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
    std::string outPath = "../.././data/mp4Reader_saveOrCompare/h264/frame_000???.jpg";

    bool parseFS = false;
    auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
    auto frameType = FrameMetadata::FrameType::H264_DATA;
    auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, true, false);
    mp4ReaderProps.fps = 24;
    mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    mp4Reader->addOutPutPin(h264ImageMetadata);

    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
    mp4Reader->addOutPutPin(mp4Metadata);

    std::vector<std::string> mImagePin;
    mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

    decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
    mp4Reader->setNext(decoder, mImagePin);

    auto conversionType = ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(conversionType)));
    decoder->setNext(colorchange);

    imageViewerSink = boost::shared_ptr<ImageViewerModule>(
        new ImageViewerModule(ImageViewerModuleProps("imageview")));
    colorchange->setNext(imageViewerSink);

    return true;
}

bool PlayMp4VideoFromBeginning::startPipeLine() {
    pipeLine.appendModule(mp4Reader);
	pipeLine.init();
    pipeLine.run_all_threaded();
    return true;
}

bool PlayMp4VideoFromBeginning::stopPipeLine() {
    pipeLine.stop();
    pipeLine.term();
    pipeLine.wait_for_all();
    return true;
}

bool PlayMp4VideoFromBeginning::flushQueuesAndSeek() {
    pipeLine.flushAllQueues();
    mp4Reader->randomSeek(1686723796848, false);
    return true;
}

int main() { 
    PlayMp4VideoFromBeginning pipelineInstance;
  if (!pipelineInstance.setUpPipeLine()) {
      std::cerr << "Failed to setup pipeline." << std::endl;
    return 1;
  }
  if (!pipelineInstance.testPipeLineForFlushQue()) {
    std::cerr << "Failed to test pipeline." << std::endl;
    return 1;
  }
  if (!pipelineInstance.testPipeLineForSeek()) {
    std::cerr << "Failed to test pipeline." << std::endl;
    return 1;
  }
  if (!pipelineInstance.startPipeLine()) {
    std::cerr << "Failed to start pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }
  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(3));
  if (!pipelineInstance.flushQueuesAndSeek()) {
    std::cerr << "Failed to flush Queues." << std::endl;
    return 1;
  }
  boost::this_thread::sleep_for(boost::chrono::seconds(5));
  // Stop the pipeline
  if (!pipelineInstance.stopPipeLine()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }
  
  return 0; // Or any success code
}
