#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "ExternalSourceModule.h"
#include "FileWriterModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include <fstream>
#include <vector>
#include "RTSPClientSrc.h"
#include "H264Metadata.h"
#include "PipeLine.h"
BOOST_AUTO_TEST_SUITE(filewritermodule_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	std::vector<std::string> Files = { "./data/testOutput/fileWriterModuleFrame_0000.jpg" , "./data/testOutput/fileWriterModuleFrame_0001.jpg", "./data/testOutput/fileWriterModuleFrame_0002.jpg", "./data/testOutput/fileWriterModuleFrame_0003.jpg" };
	Test_Utils::FileCleaner f(Files);
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

	auto frame = m1->makeFrame(readDataSize, pinId);
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

BOOST_AUTO_TEST_CASE(append)
{
	std::vector<std::string> Files = { "./data/testOutput/fileWriterModuleSample.txt" };
	Test_Utils::FileCleaner f(Files);
	unsigned int readDataSize = 0U;
	ofstream myFile("./data/testOutput/fileWriterModuleSample.txt");
	myFile << "Foo";
	myFile.close(); 
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId = m1->addOutputPin(metadata);
	auto m2 = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/fileWriterModuleSample.txt", true)));
	m1->setNext(m2);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	const char *stringToAppend = "bar";
	readDataSize = std::strlen(stringToAppend);
	auto frame = m1->makeFrame(readDataSize, pinId);
	memcpy(frame->data(), stringToAppend, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));
	m1->send(frames);
	m2->step();
	string text;
	ifstream readFile("./data/testOutput/fileWriterModuleSample.txt");
	std::getline(readFile, text);
	readFile.close();
	BOOST_TEST(text == "Foobar"); 
}

BOOST_AUTO_TEST_CASE(appendTestPattern)
{
	std::vector<std::string> Files = { "./data/testOutput/fileWriterModuleSample_0000.txt" };
	Test_Utils::FileCleaner f(Files);
	unsigned int readDataSize = 0U;
	
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId = m1->addOutputPin(metadata);
	auto m2 = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/fileWriterModuleSample_????.txt", true)));
	m1->setNext(m2);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	const char *stringToAppend = "Foo";
	readDataSize = strlen(stringToAppend);
	auto frame = m1->makeFrame(readDataSize, pinId);
	memcpy(frame->data(), stringToAppend, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));
	m1->send(frames);
	m2->step();

	string text;
	ifstream readFile("./data/testOutput/fileWriterModuleSample_0000.txt");
	std::getline(readFile, text);
	readFile.close();
	BOOST_TEST(text == "Foo");

	stringToAppend = "bar";
	readDataSize = strlen(stringToAppend);
	memcpy(frame->data(), stringToAppend, readDataSize);
	m1->send(frames);
	m2->step();
	readFile.open("./data/testOutput/fileWriterModuleSample_0000.txt");
	std::getline(readFile, text);
	readFile.close();
	BOOST_TEST(text == "Foobar");
}

BOOST_AUTO_TEST_CASE(rtsp)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	//rtsp_client_tests_data d;

	bool overlayFrames = true;
	const std::string url = "rtsp://evo-dev-apra.blub0xSecurity.com:5544/af356968-7ba2-4cee-802f-924014c6f24f";
	std::string username = "";
	std::string password = "";
	auto rtspSrc = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url, username, password)));
	auto meta = framemetadata_sp(new H264Metadata());
	rtspSrc->addOutputPin(meta);

	auto m2 = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/rtspframes/fileWriterModuleFrame_????.H264")));
	rtspSrc->setNext(m2);

	PipeLine p("test");
	p.appendModule(rtspSrc);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(15));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()