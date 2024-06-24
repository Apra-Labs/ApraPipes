
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

PlayMp4VideoFromBeginning::PlayMp4VideoFromBeginning()
    : pipeLine("PlayMp4videoFromBeginingSamplePipline") {}

bool PlayMp4VideoFromBeginning::setUpPipeLine(const std::string &videoPath) {
  // Implementation
  bool parseFS = false;
  auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
  auto frameType = FrameMetadata::FrameType::H264_DATA;
  auto mp4ReaderProps =
      Mp4ReaderSourceProps(videoPath, parseFS, 0, true, true, false);
  mp4ReaderProps.fps = 24;
  // initializing source Mp4 reader to read Mp4 video
  mMp4Reader =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  mMp4Reader->addOutPutPin(h264ImageMetadata);

  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mMp4Reader->addOutPutPin(mp4Metadata);

  std::vector<std::string> mImagePin;
  mImagePin = mMp4Reader->getAllOutputPinsByType(frameType);

  // initializing H264 decoder to decode frame in H264 format
  mDecoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));

  // Selecting an image pin of H264 data frame type is necessary because the
  // decoder processes H264 frames for decoding.
  mMp4Reader->setNext(mDecoder, mImagePin);

  //initializing conversion type module to convert YUV420 frame to RGB
  auto conversionType =
      ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
  auto metadata = framemetadata_sp(new RawImagePlanarMetadata(
      1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
  mColorchange = boost::shared_ptr<ColorConversion>(
      new ColorConversion(ColorConversionProps(conversionType)));
  mDecoder->setNext(mColorchange);

  //initializing imageViewer module as sink to show video on screen
  mImageViewerSink = boost::shared_ptr<ImageViewerModule>(
      new ImageViewerModule(ImageViewerModuleProps("imageview")));
  mColorchange->setNext(mImageViewerSink);

  return true;
}

bool PlayMp4VideoFromBeginning::startPipeLine() {
  pipeLine.appendModule(mMp4Reader);
  if (!pipeLine.init()) {
    throw AIPException(
        AIP_FATAL,
        "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    return false;
  }
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
  mMp4Reader->randomSeek(1686723796848, false);
  return true;
}
