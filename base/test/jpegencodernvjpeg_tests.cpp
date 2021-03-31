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
#include "JPEGEncoderNVJPEG.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(jpegencodernvjpeg_tests)

BOOST_AUTO_TEST_CASE(mono_1920x1080)
{
	// metadata is known
	auto width = 1920;
	auto height = 1080;
	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	auto m2 = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	copy->setNext(m2);
	auto encodedImagePin = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());
	

	
	fileReader->step();
	copy->step();
	m2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegencodernvjpeg_tests_mono_1920x1080.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	auto m2 = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	copy->setNext(m2);
	auto encodedImagePin = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());



	fileReader->step();
	copy->step();
	m2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegencodernvjpeg_tests_yuv420_640x360.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(rgb_1280x720)
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	auto m2 = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	copy->setNext(m2);
	auto encodedImagePin = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());



	fileReader->step();
	copy->step();
	m2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegencodernvjpeg_tests_frame_1280x720_rgb.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
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
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	JPEGEncoderNVJPEGProps props(stream);
	props.logHealth = true;
	auto m2 = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(props));
	copy->setNext(m2);
	auto encodedImagePin = m2->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	for (auto i = 0; i < 1; i++)
	{
		fileReader->step();
		copy->step();
		m2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_SUITE_END()
