#include <boost/test/unit_test.hpp>
#include "BayerToYUV420.h"
#include "FileWriterModule.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "FileReaderModule.h"
#include "FileSequenceDriver.h"
 
BOOST_AUTO_TEST_SUITE(BayerToYUV420_tests)
 
BOOST_AUTO_TEST_CASE(bayertoyuv420, *boost::unit_test::disabled())
{
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);
 
    auto bayertoyuv420 = boost::shared_ptr<BayerToYUV420>(new BayerToYUV420(BayerToYUV420Props())
    fileReader->setNext(bayertoyuv420);
 
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    bayertoyuv420->setNext(sink);
 
    BOOST_TEST(fileReader->init());
    BOOST_TEST(bayertoyuv420->init());
    BOOST_TEST(sink->init());
 
    fileReader->step();
    bayertoyuv420->step();
    auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
 
    Test_Utils::saveOrCompare("./data/testOutput/frame_1280x720_bayer_cc_yuv420.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
 
}
BOOST_AUTO_TEST_SUITE_END()
