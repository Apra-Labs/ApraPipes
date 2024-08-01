#include "relay_sample.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "ImageViewerModule.h"
#include "KeyboardListener.h"
#include "Logger.h"
#include "Mp4VideoMetadata.h"
#include "PipeLine.h"
#include "RTSPClientSrc.h"
#include "stdafx.h"
#include <ColorConversionXForm.h>
#include <H264Decoder.h>
#include <Mp4ReaderSource.h>
#include <boost/test/unit_test.hpp>

RelayPipeline::RelayPipeline() : pipeline("RelaySample") {}

void RelayPipeline::addRelayToRtsp(bool open) {
  rtspSource->relay(h264Decoder, open);
}

void RelayPipeline::addRelayToMp4(bool open) {
  mp4ReaderSource->relay(h264Decoder, open);
}

//bool RelayPipeline::testPipeline() {
//
//  auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
//  colorConversion->setNext(sink);
//
//  BOOST_TEST(rtspSource->init());
//  BOOST_TEST(mp4ReaderSource->init());
//  BOOST_TEST(h264Decoder->init());
//  BOOST_TEST(colorConversion->init());
//  BOOST_TEST(sink->init());
//
//  for (int i = 0; i <= 10; i++) {
//    mp4ReaderSource->step();
//    h264Decoder->step();
//  }
//  colorConversion->step();
//
//  auto frames = sink->pop();
//  frame_sp outputFrame = frames.cbegin()->second;
//  Test_Utils::saveOrCompare(
//      "../.././data/frame_from_mp4.raw",
//      const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())),
//      outputFrame->size(), 0);
//
//  for (int i = 0; i <= 10; i++) {
//    rtspSource->step();
//    h264Decoder->step();
//  }
//  colorConversion->step();
//
//  frames = sink->pop();
//  outputFrame = frames.cbegin()->second;
//  Test_Utils::saveOrCompare(
//      "../.././data/frame_from_rtsp.raw",
//      const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())),
//      outputFrame->size(), 0);
//
//  return true;
//}

bool RelayPipeline::setupPipeline(const std::string &rtspUrl, const std::string &mp4VideoPath) {
  // RTSP
  auto url = std::string(rtspUrl);
  rtspSource = boost::shared_ptr<RTSPClientSrc>(
      new RTSPClientSrc(RTSPClientSrcProps(url, "", "")));
  auto rtspMetaData = framemetadata_sp(new H264Metadata(1280, 720));
  rtspSource->addOutputPin(rtspMetaData);

  // MP4
  bool parseFS = false;
  auto h264ImageMetadata = framemetadata_sp(new H264Metadata(1280, 720));
  auto frameType = FrameMetadata::FrameType::H264_DATA;
  auto mp4ReaderProps =
      Mp4ReaderSourceProps(mp4VideoPath, parseFS, 0, true, true, false);
  mp4ReaderProps.fps = 9;
  mp4ReaderSource =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  mp4ReaderSource->addOutPutPin(h264ImageMetadata);
  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mp4ReaderSource->addOutPutPin(mp4Metadata);
  std::vector<std::string> mImagePin;
  mImagePin = mp4ReaderSource->getAllOutputPinsByType(frameType);

  h264Decoder =
      boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
  rtspSource->setNext(h264Decoder);
  mp4ReaderSource->setNext(h264Decoder, mImagePin);

  colorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(
      ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
  h264Decoder->setNext(colorConversion);

  imageViewer = boost::shared_ptr<ImageViewerModule>(
      new ImageViewerModule(ImageViewerModuleProps("Relay Sample")));
  colorConversion->setNext(imageViewer);
  return true;
}

bool RelayPipeline::startPipeline() {
  pipeline.appendModule(rtspSource);
  pipeline.appendModule(mp4ReaderSource);
  pipeline.init();
  pipeline.run_all_threaded();
  addRelayToMp4(false);
  return true;
}

bool RelayPipeline::stopPipeline() {
  pipeline.stop();
  pipeline.term();
  pipeline.wait_for_all();
  return true;
}

