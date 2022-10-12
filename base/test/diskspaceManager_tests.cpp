#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "DiskspaceManager.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "Module.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(diskspaceManager_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	//auto diskMan = boost::shared_ptr<DiskspaceManager>(new DiskspaceManager(DiskspaceManagerProps(150000,190000, "C://Users//Vinayak//Desktop//Work//RedBull",".*[t][x][t]")));
	auto diskMan = boost::shared_ptr<DiskspaceManager>(new DiskspaceManager(DiskspaceManagerProps(600, 900, "C://Users//Vinayak//Desktop//Work//RedBull 2.0//October", ".h264"))); 
	source->setNext(diskMan);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());
	auto frame = source->makeFrame(1023, pinId);
	frame_container frames;
	frames.insert(make_pair(pinId, frame));
	source->send(frames);
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(source);
	//p->init();
	p->run_all_threaded();
	//Test_Utils::sleep_for_seconds(20);
	Test_Utils::sleep_for_seconds(600);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}
BOOST_AUTO_TEST_SUITE_END()