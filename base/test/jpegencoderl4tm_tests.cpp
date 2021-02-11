#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "JPEGEncoderL4TM.h"
#include "FileReaderModule.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "test_utils.h"
#include <fstream>

BOOST_AUTO_TEST_SUITE(jpegencoderl4tm_tests)

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic)
{
	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_rgb)
{
	// metadata is known
	auto width = 1280;
	auto height = 720;
	auto fileSize = width*height*3;
	unsigned char *in_buf = new unsigned char[fileSize];

	auto in_file = new std::ifstream("./data/frame_1280x720_rgb.raw");
	in_file->read((char *)in_buf, fileSize);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());	
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegencoderl4tm_frame_1280x720_rgb.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_scale)
{
	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.scale = 0.125;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm_scale_0.125.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0); 
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_rgb_perf, * boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 1280;
	auto height = 720;
	FileReaderModuleProps fileReaderProps("./data/frame_1280x720_rgb.raw", 0, -1, 4*1024*1024);
	fileReaderProps.fps = 1000;
	auto m1 = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));	
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps encoderProps;
	encoderProps.logHealth = true;
	encoderProps.logHealthFrequency = 100;
	auto m2 = boost::shared_ptr<Module>(new JPEGEncoderL4TM(encoderProps));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto m3 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	m2->setNext(m3);

	PipeLine p("test");
	p.appendModule(m1);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_perf, * boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.logHealth = true;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	for (auto i = 0; i < 10000; i++)
	{
		m1->send(frames);
		m2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_perf_scale, * boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.logHealth = true;
	props.scale = 0.25;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	for (auto i = 0; i < 10000; i++)
	{
		m1->send(frames);
		m2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_2, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_notmultipleof32, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img_orig = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img;
	cv::resize(img_orig, img, cv::Size(240, 60));
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);

	try
	{
		m2->step();
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_NOTIMPLEMENTED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_notmultipleof32_2, * boost::unit_test::disabled())
{
	// metadata is known
	auto img_orig = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img;
	cv::resize(img_orig, img, cv::Size(240, 60));
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	try
	{
		m1->setNext(m2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_channels_2, * boost::unit_test::disabled())
{
	// metadata is known
	auto img = cv::imread("./data/frame.jpg");
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	try
	{
		m1->setNext(m2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_channels, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img = cv::imread("./data/frame.jpg");
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);

	try
	{
		m2->step();
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_NOTIMPLEMENTED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_SUITE_END()
