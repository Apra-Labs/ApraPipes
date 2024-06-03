#include "GenerateThumbnailsPipeline.h"
#include "test_utils.h"
#include <boost/test/unit_test.hpp>
#include <chrono>

BOOST_AUTO_TEST_SUITE(generateThumbnails)

BOOST_AUTO_TEST_CASE(generateThumbnails_from_mp4) {
  auto generateThumbnailPipeline =
      boost::shared_ptr<GenerateThumbnailsPipeline>(
          new GenerateThumbnailsPipeline());
  std::string videoPath = "C:/APRA/fork/ApraPipes/data/Mp4_videos/h264_video_metadata/20230514/0011/1686723796848.mp4";
  std::string outFolderPath = "C:/APRA/fork/ApraPipes/samples/create_thumbnail_from_mp4_video/data/generated_thumbnail/thumbnail_????.jpg";
  generateThumbnailPipeline->setUpPipeLine(videoPath, outFolderPath);
  generateThumbnailPipeline->startPipeLine();

  boost::this_thread::sleep_for(boost::chrono::seconds(5));

  const uint8_t *pReadDataTest = nullptr;
  unsigned int readDataSizeTest = 0U;

  BOOST_TEST(Test_Utils::readFile("C:/APRA/fork/ApraPipes/samples/create_thumbnail_from_mp4_video/data/test_thumbnail/sample_thumbnail.jpg",
                                  pReadDataTest, readDataSizeTest));
  Test_Utils::saveOrCompare(
      "C:/APRA/fork/ApraPipes/samples/create_thumbnail_from_mp4_video/data/generated_thumbnail/thumbnail_0000.jpg", pReadDataTest,
      readDataSizeTest, 0);

  generateThumbnailPipeline->stopPipeLine();
}

BOOST_AUTO_TEST_SUITE_END()