#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "ExternalSourceModule.h"
#include "FileWriterModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(filewritermodule_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	const uint8_t* pReadData = nullptr;
	unsigned int readDataSize = 0U;
	BOOST_TEST(Test_Utils::readFile("./data/mono.jpg", pReadData, readDataSize));
	
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));	
	auto pinId = m1->addOutputPin(metadata);

	Test_Utils::createDirIfNotExist("./data/testOutput/mono.jpg");
	auto m2 = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/fileWriterModuleFrame_????.jpg")));
	m1->setNext(m2);
	
	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());

	auto frame = m1->makeFrame(readDataSize, metadata);
	memcpy(frame->data(), pReadData, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));
	
	for (auto i = 0; i < 4; i++)
	{
		m1->send(frames);
		m2->step();

		const uint8_t* pReadDataTest = nullptr;
		unsigned int readDataSizeTest = 0U;
		BOOST_TEST(Test_Utils::readFile("./data/testOutput/fileWriterModuleFrame_000" + to_string(i) + ".jpg", pReadDataTest, readDataSizeTest));

		Test_Utils::saveOrCompare("./data/testOutput/mono.jpg", pReadDataTest, readDataSizeTest, 0);

		delete[] pReadDataTest;
	}

	delete[] pReadData;
}

BOOST_AUTO_TEST_SUITE_END()