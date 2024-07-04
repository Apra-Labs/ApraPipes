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
		outFile = string("./data/testOutput/bunny_????.h264");
		Test_Utils::FileCleaner fc;
		fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
	}
	string outFile;
	string empty;
};

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	rtsp_client_tests_data d;

	//drop bunny/mp4 into evostream folder, 
	//also set it up for RTSP client authentication as shown here: https://sites.google.com/apra.in/development/home/evostream/rtsp-authentication?authuser=1
	auto url=string("rtsp://10.102.10.75/axis-media/media.amp?resolution=1280x720"); 
	
	auto m = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	m->addOutputPin(meta);
	
	//filewriter for saving output
	auto fw = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(d.outFile)));
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
