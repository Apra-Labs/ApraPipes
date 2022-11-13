#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "RTSPClientSrc.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "FileWriterModule.h"
using namespace std;

BOOST_AUTO_TEST_SUITE(rtsp_client_tests)

struct rtsp_client_tests_data {
	rtsp_client_tests_data()
	{
		outFile = string("./data/testOutput/bunny.h264");
		Test_Utils::FileCleaner fc;
		fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
	}
	string outFile;
	string empty;
};

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	rtsp_client_tests_data d;
	auto url=string("rtsp://127.0.0.1:5544/vod/mp4:bunny.mp4"); //drop bunny/mp4 into evostream folder
	
	auto m = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	m->addOutputPin(framemetadata_sp(new H264Metadata()));
	
	//filewriter for saving output
	auto fw = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(d.outFile, true)));
	m->setNext(fw);


	BOOST_TEST(m->init());
	BOOST_TEST(fw->init());

	for (int i = 0;i<1000; i++)
	{
		BOOST_TEST(m->step());
		BOOST_TEST(fw->step());
	}

	m->term();
	fw->term();

}

BOOST_AUTO_TEST_SUITE_END()
