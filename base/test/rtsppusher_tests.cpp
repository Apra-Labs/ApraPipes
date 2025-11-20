#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <memory>
#include <thread>
#include <chrono>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "RTSPPusher.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(rtsppusher_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	std::string dataPath = Test_Utils::getArgValue("data", "./data/h264data/frame.bin"); //works
	//std::string dataPath = Test_Utils::getArgValue("data", "./data/h264data/9bfe"); //works
	//std::string dataPath = Test_Utils::getArgValue("data", "./data/h264data/84e9");  //works
	//std::string dataPath = Test_Utils::getArgValue("data", "./data/h264data/769d-1");  //works
	//std::string dataPath = Test_Utils::getArgValue("data", "./data/h264data/769d-rec");  //works
	
	std::string rtspServer = Test_Utils::getArgValue("rtspserver", "rtsp://10.102.10.81:5544");
	
	auto encoderTargetKbpsStr = Test_Utils::getArgValue("bitrate", "4096");
	uint32_t encoderTargetKbps = atoi(encoderTargetKbpsStr.c_str());

	FileReaderModuleProps fileReaderProps(dataPath);
	fileReaderProps.fps = 30;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	fileReader->addOutputPin(metadata);

	RTSPPusherProps sinkProps(rtspServer, "hola");
	sinkProps.encoderTargetKbps = encoderTargetKbps;
	auto sink = std::shared_ptr<Module>(new RTSPPusher(sinkProps));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(500000000000000));

	LOG_INFO << "STOPPING";

	p.stop();
	p.term();
	LOG_INFO << "WAITING";
	p.wait_for_all();
	LOG_INFO << "TEST DONE";
}

BOOST_AUTO_TEST_SUITE_END()
