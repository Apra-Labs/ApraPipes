#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "RotateNPPI.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(rotatenppi_tests)

void test(std::string filename, int width, int height, ImageMetadata::ImageType imageType, int type, int depth, double angle)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/" + filename + ".raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, imageType, type, 0, depth, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new RotateNPPI(RotateNPPIProps(stream, angle)));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());

	fileReader->step();
	copy1->step();
	m2->step();
	copy2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	auto outFilename = "./data/testOutput/rotatenppi_tests_" + filename + "_" + std::to_string(angle) + ".raw";
	Test_Utils::saveOrCompare(outFilename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);

	BOOST_TEST(fileReader->term());
	BOOST_TEST(copy1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(copy2->term());
	BOOST_TEST(m3->term());
}

BOOST_AUTO_TEST_CASE(mono_8U_90_cc)
{
	test("mono_1920x1080", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, CV_8U, 90);
}

BOOST_AUTO_TEST_CASE(mono_8U_90_c)
{
	test("mono_1920x1080", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, CV_8U, -90);
}

BOOST_AUTO_TEST_CASE(mono_16U_90_cc)
{
	test("depth_1280x720", 1280, 720, ImageMetadata::ImageType::MONO, CV_16UC1, CV_16U, 90);
}

BOOST_AUTO_TEST_CASE(mono_16U_90_c)
{
	test("depth_1280x720", 1280, 720, ImageMetadata::ImageType::MONO, CV_16UC1, CV_16U, -90);
}

BOOST_AUTO_TEST_CASE(rgb_8U_90_cc)
{
	test("frame_1280x720_rgb", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, CV_8U, 90);
}

BOOST_AUTO_TEST_CASE(rgb_8U_90_c)
{
	test("frame_1280x720_rgb", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, CV_8U, -90);
}

BOOST_AUTO_TEST_CASE(perf, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/4k.yuv")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new RotateNPPI(RotateNPPIProps(stream, 90)));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());

	for (auto i = 0; i < 1; i++)
	{
		fileReader->step();
		copy1->step();
		m2->step();
		copy2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_SUITE_END()
