
#include "PlayMp4VideoFromBeginning.h"
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
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include <boost/filesystem.hpp>

PlayMp4VideoFromBeginning::PlayMp4VideoFromBeginning() : pipeLine("pipeline") {}

bool PlayMp4VideoFromBeginning::setUpPipeLine(const std::string &videoPath) {
  // Implementation
  bool parseFS = false;
  auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
  auto frameType = FrameMetadata::FrameType::H264_DATA;
  auto mp4ReaderProps =
      Mp4ReaderSourceProps(videoPath, parseFS, 0, true, true, false);
  mp4ReaderProps.fps = 24;
  mp4Reader =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  mp4Reader->addOutPutPin(h264ImageMetadata);

  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mp4Reader->addOutPutPin(mp4Metadata);

  std::vector<std::string> mImagePin;
  mImagePin = mp4Reader->getAllOutputPinsByType(frameType);

  decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
  mp4Reader->setNext(decoder, mImagePin);

  auto conversionType =
      ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
  auto metadata = framemetadata_sp(new RawImagePlanarMetadata(
      1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
  colorchange = boost::shared_ptr<ColorConversion>(
      new ColorConversion(ColorConversionProps(conversionType)));
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
