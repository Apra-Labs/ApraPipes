#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "RTSPPusher.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "RTSPClientSrc.h"
#include "H264Metadata.h"

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
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	fileReader->addOutputPin(metadata);

	RTSPPusherProps sinkProps(rtspServer, "hola");
	sinkProps.encoderTargetKbps = encoderTargetKbps;
	auto sink = boost::shared_ptr<RTSPPusher>(new RTSPPusher(sinkProps));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(500000000000000));

	LOG_INFO << "STOPPING";

	p.stop();
	p.term();
	LOG_INFO << "WAITING";
	p.wait_for_all();
	LOG_INFO << "TEST DONE";
}

struct rtsp_client_tests_data {
	rtsp_client_tests_data()
	{
		outFile = string("./data/testOutput/bunny_????.h264");
		Test_Utils::FileCleaner fc;
		fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
	}
	string outFile;
	string empty;
};

BOOST_AUTO_TEST_CASE(rtsp_fetcher_and_pusher, *boost::unit_test::disabled())
{	
	rtsp_client_tests_data d;
	auto url=string("rtsp://root:m4m1g0@10.102.10.77/axis-media/media.amp?resolution=1280x720"); 
	std::string rtspServer = Test_Utils::getArgValue("rtspserver", "rtsp://10.102.10.197:5544");
	auto encoderTargetKbpsStr = Test_Utils::getArgValue("bitrate", "4096");
	uint32_t encoderTargetKbps = atoi(encoderTargetKbpsStr.c_str());

	auto src = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	src->addOutputPin(meta);

	RTSPPusherProps sinkProps(rtspServer, "hola");
	sinkProps.encoderTargetKbps = encoderTargetKbps;
	auto sink = boost::shared_ptr<RTSPPusher>(new RTSPPusher(sinkProps));
	src->setNext(sink);

	PipeLine p("test");
	p.appendModule(src);
	bool pipeLineInitStatus = p.init();
	// BOOST_TEST(p.init());
	LOG_ERROR << "Pipeline Init Status " << pipeLineInitStatus;
	if (pipeLineInitStatus)
	{
		p.run_all_threaded();

		boost::this_thread::sleep_for(boost::chrono::seconds(5));
		// p.pause();
		LOG_ERROR << "Will Pause the Stream";
		sink->setPausedState(true);
		boost::this_thread::sleep_for(boost::chrono::seconds(15));
		LOG_ERROR << "Will Play the Stream";
		sink->setPausedState(false);
		boost::this_thread::sleep_for(boost::chrono::seconds(5));
		LOG_ERROR << "Will Pause the Stream";
		sink->setPausedState(true);
		boost::this_thread::sleep_for(boost::chrono::seconds(15));
		LOG_ERROR << "Will Play the Stream";
		sink->setPausedState(false);
		boost::this_thread::sleep_for(boost::chrono::seconds(500000000000000));

		LOG_INFO << "STOPPING";

		p.stop();
		p.term();
		LOG_INFO << "WAITING";
		p.wait_for_all();
		LOG_INFO << "TEST DONE";
	}
}
BOOST_AUTO_TEST_SUITE_END()
