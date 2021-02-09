#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "RTSPPusher.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(rtsppusher_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	FileReaderModuleProps fileReaderProps("./data/ArgusCamera");
	fileReaderProps.fps = 30;
	fileReaderProps.maxFileSize = 200*1024;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<Module>(new RTSPPusher(RTSPPusherProps("rtsp://10.102.10.129:5544", "hola")));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(500000000000000));

	LOG_ERROR << "STOPPING";

	p.stop();
	p.term();
	LOG_ERROR << "WAITING";
	p.wait_for_all();
	LOG_ERROR << "TEST DONE";
}

BOOST_AUTO_TEST_SUITE_END()
