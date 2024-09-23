#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "RTSPClientSrc.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "FileWriterModule.h"
#include "Mp4WriterSink.h"
#include "PipeLine.h"

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

BOOST_AUTO_TEST_CASE(recording24x7, *boost::unit_test::disabled())
{
	rtsp_client_tests_data d;
	std::string mp4Writer_247_Path = "./TestVideos/";
	uint32_t mp4Writer_247_chunk = 1; // Default
	uint32_t mp4Writer_247_sync = 2;
	uint32_t mp4Writer_247_fps = 24;

	// drop bunny/mp4 into evostream folder,
	// also set it up for RTSP client authentication as shown here: https://sites.google.com/apra.in/development/home/evostream/rtsp-authentication?authuser=1
	auto url = string("rtsp://10.102.10.79/axis-media/media.amp?resolution=1280x720");

	auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	source->addOutputPin(meta);

	// Mp4-Writer 24/7
	Mp4WriterSinkProps mp4Writer_247_Props = Mp4WriterSinkProps(mp4Writer_247_chunk, mp4Writer_247_sync, mp4Writer_247_fps, mp4Writer_247_Path);
	// mp4Writer_247_Props.logHealth = true;
	// mp4Writer_247_Props.logHealthFrequency = 100;
	boost::shared_ptr<Mp4WriterSink> mp4writer_247 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4Writer_247_Props));
	source->setNext(mp4writer_247);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("NVR_Core_Pipeline"));
	p->appendModule(source);
	bool pipelineInit = p->init();
	if (!pipelineInit)
	{
		LOG_ERROR << "Core pipeline init failed. RTSP URL most likely the problem.";
	}
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(UINT32_MAX));
	p->stop();
	p->term();
	p->wait_for_all();
}


BOOST_AUTO_TEST_CASE(getSetProps, *boost::unit_test::disabled())
{
	rtsp_client_tests_data d;
	std::string mp4Writer_247_Path = "./TestVideos/";
	uint32_t mp4Writer_247_chunk = 1; // Default
	uint32_t mp4Writer_247_sync = 2;
	uint32_t mp4Writer_247_fps = 24;

	auto url = string("rtsp://10.102.10.79/axis-media/media.amp?resolution=1280x720");

	auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	source->addOutputPin(meta);

	Mp4WriterSinkProps mp4Writer_247_Props = Mp4WriterSinkProps(mp4Writer_247_chunk, mp4Writer_247_sync, mp4Writer_247_fps, mp4Writer_247_Path);
	boost::shared_ptr<Mp4WriterSink> mp4writer_247 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4Writer_247_Props));
	source->setNext(mp4writer_247);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("NVR_Core_Pipeline"));
	p->appendModule(source);
	bool pipelineInit = p->init();
	if (!pipelineInit)
	{
		LOG_ERROR << "Core pipeline init failed. RTSP URL most likely the problem.";
	}
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	RTSPClientSrcProps currProps = source->getProps();

	auto updatedUrl = string("rtsp://root:m4m1g0@10.102.10.56/axis-media/media.amp?resolution=1280x720");
	currProps.rtspURL = updatedUrl;
	source->setProps(currProps);
	boost::this_thread::sleep_for(boost::chrono::seconds(UINT32_MAX));
	p->stop();
	p->term();
	p->wait_for_all();
}


BOOST_AUTO_TEST_CASE(getSetFilePathChange, *boost::unit_test::disabled())
{
	rtsp_client_tests_data d;
	std::string mp4Writer_247_Path = "./cam79/";
	std::string mp4Writer_247_Path_New = "./cam79_new/";
	uint32_t mp4Writer_247_chunk = 1; // Default
	uint32_t mp4Writer_247_sync = 2;
	uint32_t mp4Writer_247_fps = 24;

	auto url = string("rtsp://10.102.10.79/axis-media/media.amp?resolution=1280x720");

	auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	source->addOutputPin(meta);

	Mp4WriterSinkProps mp4Writer_247_Props = Mp4WriterSinkProps(mp4Writer_247_chunk, mp4Writer_247_sync, mp4Writer_247_fps, mp4Writer_247_Path);
	boost::shared_ptr<Mp4WriterSink> mp4writer_247 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4Writer_247_Props));
	source->setNext(mp4writer_247);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("NVR_Core_Pipeline"));
	p->appendModule(source);
	bool pipelineInit = p->init();

	if (!pipelineInit)
	{
		LOG_ERROR << "Core pipeline init failed. RTSP URL most likely the problem.";
	}
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(19));
	auto currMp4Props = mp4writer_247->getProps();
	currMp4Props.baseFolder = mp4Writer_247_Path_New;
	mp4writer_247->setProps(currMp4Props);
	boost::this_thread::sleep_for(boost::chrono::seconds(UINT32_MAX));
	p->stop();
	p->term();
	p->wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
