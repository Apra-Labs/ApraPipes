
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
#include "ImageViewerModule.h"


PlayMp4VideoFromBeginning::PlayMp4VideoFromBeginning() : pipeLine("pipeline") {

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

    auto decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
    mp4Reader->setNext(decoder, mImagePin);

    auto conversionType = ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(conversionType)));
    decoder->setNext(colorchange);

    sink = boost::shared_ptr<ImageViewerModule>(
        new ImageViewerModule(ImageViewerModuleProps("imageview")));
    colorchange->setNext(sink);

    pipeLine.appendModule(mp4Reader);
	pipeLine.init();
    return true;
}

bool PlayMp4VideoFromBeginning::startPipeLine() {
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
