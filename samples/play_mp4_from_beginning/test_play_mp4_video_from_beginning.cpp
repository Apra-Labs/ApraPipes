#include "Frame.h"
#include "FrameContainerQueue.h"
#include "PlayMp4VideoFromBeginning.h"
#include "test_utils.h"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(start_video_from_beginning)
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

BOOST_AUTO_TEST_CASE(play_mp4_from_beginning_flush_queue_test) {
  auto playMp4VideoFromBeginning = boost::shared_ptr<PlayMp4VideoFromBeginning>(
      new PlayMp4VideoFromBeginning());
  std::string videoPath = "C:/APRA/fork/ApraPipes/data/Mp4_videos/"
                          "h264_video_metadata/20230514/0011/1686723796848.mp4";
  playMp4VideoFromBeginning->setUpPipeLine(videoPath);

  auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
  playMp4VideoFromBeginning->colorchange->setNext(sink);
  BOOST_CHECK(playMp4VideoFromBeginning->mp4Reader->init());
  BOOST_CHECK(playMp4VideoFromBeginning->decoder->init());
  BOOST_CHECK(playMp4VideoFromBeginning->colorchange->init());
  BOOST_CHECK(sink->init());

  for (int i = 0; i <= 20; i++) {
    playMp4VideoFromBeginning->mp4Reader->step();
    playMp4VideoFromBeginning->decoder->step();
  }
  playMp4VideoFromBeginning->colorchange->step();

  auto frames = sink->pop();
  auto frame = frames.size();
  auto sinkQue = sink->getQue();
  BOOST_CHECK_EQUAL(frames.size(), 1);
  sinkQue->flush();
  BOOST_CHECK_EQUAL(sinkQue->size(), 0);
  frame_sp outputFrame = frames.cbegin()->second;
  Test_Utils::saveOrCompare(
      "./data/mp4Reader_saveOrCompare/h264/testplay.raw",
      const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())),
      outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(play_mp4_from_beginning_seek_test) {
  auto playMp4VideoFromBeginning = boost::shared_ptr<PlayMp4VideoFromBeginning>(
      new PlayMp4VideoFromBeginning());
  std::string videoPath = "C:/APRA/fork/ApraPipes/data/Mp4_videos/"
                          "h264_video_metadata/20230514/0011/1686723796848.mp4";
  playMp4VideoFromBeginning->setUpPipeLine(videoPath);
  auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
  playMp4VideoFromBeginning->mp4Reader->setNext(sink);

  BOOST_CHECK(playMp4VideoFromBeginning->mp4Reader->init());
  BOOST_CHECK(playMp4VideoFromBeginning->decoder->init());
  BOOST_CHECK(playMp4VideoFromBeginning->colorchange->init());
  BOOST_CHECK(sink->init());

  playMp4VideoFromBeginning->mp4Reader->step();

  auto frames = sink->pop();
  auto imgFrame = frames.begin()->second;

  uint64_t skipTS = 1686723796848;
  playMp4VideoFromBeginning->mp4Reader->randomSeek(skipTS, false);
  playMp4VideoFromBeginning->mp4Reader->step();
  frames = sink->pop();
  imgFrame = frames.begin()->second;
  BOOST_TEST(imgFrame->timestamp == 1686723796848);
  LOG_INFO << "Found next available frame " << imgFrame->timestamp - skipTS
           << " msecs later from skipTS";
}

BOOST_AUTO_TEST_SUITE_END()