#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "JPEGDecoderL4TM.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(jpegdecoderl4tm_tests)

BOOST_AUTO_TEST_CASE(jpegdecoderl4tm_basic, * boost::unit_test::disabled())
{	
	const uint8_t* pReadData = nullptr;
	unsigned int readDataSize = 0U;
	BOOST_TEST(Test_Utils::readFile("./data/frame.jpg", pReadData, readDataSize));
	
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));	
	auto encodedImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<Module>(new JPEGDecoderL4TM());
	m1->setNext(m2);
	auto rawImageMetadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m2->addOutputPin(rawImageMetadata);
	
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3); 

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
	memcpy(encodedImageFrame->data(), pReadData, readDataSize);

	frame_container frames;
	frames.insert(make_pair(encodedImagePin, encodedImageFrame));

	auto j = 0;
	while (true)
	{
		using sys_clock = std::chrono::system_clock;
		sys_clock::time_point frame_begin = sys_clock::now();
		for (int i = 0; i < 1000; i++)
		{
			m1->send(frames);
			m2->step();
			m3->pop();
		}
		sys_clock::time_point frame_end = sys_clock::now();
		std::chrono::nanoseconds frame_len = frame_end - frame_begin;
		LOG_ERROR << "loopindex<" << j << "> timeelapsed<" << 1.0*frame_len.count() / (1000000000.0) << "> fps<" << 1000 / (1.0*frame_len.count() / (1000000000.0)) << ">";
		j++;

		if (j == 10)
		{
			break;
		}
	}
	
	delete[] pReadData;
}

BOOST_AUTO_TEST_CASE(jpegdecoderl4tm_rgb)
{		
	const uint8_t* pReadData = nullptr;
	unsigned int readDataSize = 0U;
	BOOST_TEST(Test_Utils::readFile("./data/frame.jpg", pReadData, readDataSize));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<Module>(new JPEGDecoderL4TM());
	m1->setNext(m2);
	auto decoderMetadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m2->addOutputPin(decoderMetadata);
		
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
	memcpy(encodedImageFrame->data(), pReadData, readDataSize);

	frame_container frames;
	frames.insert(make_pair(encodedImagePin, encodedImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(rawImagePin) != frames.end()));
	auto rawImageFrame = frames[rawImagePin];

	auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(decoderMetadata);
	BOOST_TEST(rawImageMetadata->getWidth() == 1920);
	BOOST_TEST(rawImageMetadata->getHeight() == 454);
	BOOST_TEST(rawImageMetadata->getChannels() == 1);
	Test_Utils::saveOrCompare( "./data/testOutput/frame.raw", (const uint8_t*) rawImageFrame->data(), rawImageMetadata->getDataSize(), 0);

	delete[] pReadData;
}

BOOST_AUTO_TEST_CASE(jpegdecoderl4tm_mono)
{
	const uint8_t* pReadData = nullptr;
	unsigned int readDataSize = 0U;
	BOOST_TEST(Test_Utils::readFile("./data/mono.jpg", pReadData, readDataSize));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<Module>(new JPEGDecoderL4TM());
	m1->setNext(m2);
	auto decoderMetadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m2->addOutputPin(decoderMetadata);
		
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
	memcpy(encodedImageFrame->data(), pReadData, readDataSize);

	frame_container frames;
	frames.insert(make_pair(encodedImagePin, encodedImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(rawImagePin) != frames.end()));
	auto rawImageFrame = frames[rawImagePin];
	
	auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(decoderMetadata);
	BOOST_TEST(rawImageMetadata->getWidth() == 2048);
	BOOST_TEST(rawImageMetadata->getHeight() == 1536);
	BOOST_TEST(rawImageMetadata->getChannels() == 1);
	Test_Utils::saveOrCompare("./data/testOutput/mono.raw", (const uint8_t*)rawImageFrame->data(), rawImageMetadata->getDataSize(), 0);
	

	delete[] pReadData;
}

BOOST_AUTO_TEST_SUITE_END()