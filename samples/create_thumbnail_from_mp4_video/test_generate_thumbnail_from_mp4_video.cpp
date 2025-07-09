#include "GenerateThumbnailsPipeline.h"
 
#include "../../base/test/test_utils.h"

#include <boost/test/unit_test.hpp>
#include <chrono>

BOOST_AUTO_TEST_SUITE(generateThumbnails)

BOOST_AUTO_TEST_CASE(generateThumbnails_from_mp4) {
  auto generateThumbnailPipeline =
      boost::shared_ptr<GenerateThumbnailsPipeline>(
          new GenerateThumbnailsPipeline());
  std::string videoPath = "../../data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
  std::string outFolderPath = "data/generated_thumbnail/thumbnail_????.jpg";
  BOOST_CHECK_NO_THROW(generateThumbnailPipeline->setUpPipeLine(videoPath, outFolderPath));
  BOOST_CHECK_NO_THROW(generateThumbnailPipeline->startPipeLine());

  boost::this_thread::sleep_for(boost::chrono::seconds(5));

  const uint8_t *pReadDataTest = nullptr;
  unsigned int readDataSizeTest = 0U;

  BOOST_TEST(Test_Utils::readFile("data/test_thumbnail/sample_thumbnail.jpg",
                                  pReadDataTest, readDataSizeTest));
  Test_Utils::saveOrCompare(
      "data/generated_thumbnail/thumbnail_0000.jpg", pReadDataTest,
      readDataSizeTest, 0);

  generateThumbnailPipeline->stopPipeLine();
}

BOOST_AUTO_TEST_SUITE_END()