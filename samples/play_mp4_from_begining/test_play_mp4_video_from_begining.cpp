#include <boost/test/unit_test.hpp>

#include "PlayMp4VideoFromBeginning.h"

BOOST_AUTO_TEST_SUITE(start_video_from_beginning)

BOOST_AUTO_TEST_CASE(start_from_beginning)
{
  auto playMp4VideoFromBeginning = boost::shared_ptr<PlayMp4VideoFromBeginning>(new PlayMp4VideoFromBeginning());
  playMp4VideoFromBeginning->setUpPipeLine();
  playMp4VideoFromBeginning->startPipeLine();

  boost::this_thread::sleep_for(boost::chrono::seconds(10));

  playMp4VideoFromBeginning->stopPipeLine();
}

BOOST_AUTO_TEST_SUITE_END()