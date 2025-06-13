#include "relay_sample.h"
#include "test_utils.h"
#include <boost/test/unit_test.hpp>
#include <chrono>
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(relay_sample)

BOOST_AUTO_TEST_CASE(relaySample) {
  RelayPipeline relayPipeline;
  std::string rtspUrl = "rtsp://root:m4m1g0@10.102.10.75/axis-media/media.amp?resolution=1280x720";
  std::string mp4VideoPath = "data/1714992199120.mp4";
  relayPipeline.setupPipeline(rtspUrl, mp4VideoPath);


  auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
  relayPipeline.colorConversion->setNext(sink);

  BOOST_TEST(relayPipeline.rtspSource->init());
  BOOST_TEST(relayPipeline.mp4ReaderSource->init());
  BOOST_TEST(relayPipeline.h264Decoder->init());
  BOOST_TEST(relayPipeline.colorConversion->init());
  BOOST_TEST(sink->init());

  for (int i = 0; i <= 10; i++) {
    relayPipeline.mp4ReaderSource->step();
    relayPipeline.h264Decoder->step();
  }
  relayPipeline.colorConversion->step();

  auto frames = sink->pop();
  frame_sp outputFrame = frames.cbegin()->second;
  Test_Utils::saveOrCompare(
      "../.././data/frame_from_mp4.raw",
      const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())),
      outputFrame->size(), 0);

  for (int i = 0; i <= 10; i++) {
    relayPipeline.rtspSource->step();
    relayPipeline.h264Decoder->step();
  }
  relayPipeline.colorConversion->step();

  frames = sink->pop();
  outputFrame = frames.cbegin()->second;
  Test_Utils::saveOrCompare(
      "../.././data/frame_from_rtsp.raw",
      const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())),
      outputFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
