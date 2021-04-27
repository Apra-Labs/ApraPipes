#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "RotateCV.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(rotatecv_tests)

void test(std::string filename, int width, int height, ImageMetadata::ImageType imageType, int type, int depth, double angle)
{

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/" + filename + ".raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width * 3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto m1 = boost::shared_ptr<Module>(new RotateCV(RotateCVProps(angle)));
	fileReader->setNext(m1);

	auto outputPinId = m1->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m1->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(m1->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	m1->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	auto outFilename = "./data/testOutput/rotatecv_tests_" + filename + "_" + std::to_string(angle) + ".raw";
	Test_Utils::saveOrCompare(outFilename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(rgb_8U_90_c)
{
	test("frame_1280x720_rgb", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, CV_8U, 90);
}

BOOST_AUTO_TEST_SUITE_END()
