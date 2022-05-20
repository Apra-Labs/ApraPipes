#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "FileWriterModule.h"
#include "test_utils.h"
#include "DummyDMASource.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "EglRenderer.h"
BOOST_AUTO_TEST_SUITE(dummydma_test)

BOOST_AUTO_TEST_CASE(mono1_1920x960)
{
	auto dummy1 = boost::shared_ptr<DummyDMASource>(new DummyDMASource(DummyDMASourceProps("/home/developer/ApraPipes/data/frame.jpg", 1280, 720)));

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	dummy1->setNext(statSink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(dummy1);
	p->init();
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	p->stop();
	p->term();
	p->wait_for_all();
}

BOOST_AUTO_TEST_CASE(renderPipeline)
{
	auto dummy1 = boost::shared_ptr<DummyDMASource>(new DummyDMASource(DummyDMASourceProps("./data/frame.jpg", 1920, 454)));

	auto rendererProps1 = EglRendererProps(0, 0, 400, 400);
	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(rendererProps1));
	dummy1->setNext(sink1);

	auto dummy2 = boost::shared_ptr<DummyDMASource>(new DummyDMASource(DummyDMASourceProps("./data/frame.jpg", 1920, 454)));

	auto rendererProps2 = EglRendererProps(0, 400, 400, 400);
	auto sink2 = boost::shared_ptr<Module>(new EglRenderer(rendererProps2));
	dummy2->setNext(sink2);

	auto dummy3 = boost::shared_ptr<DummyDMASource>(new DummyDMASource(DummyDMASourceProps("./data/frame.jpg", 1920, 454)));

	auto rendererProps3 = EglRendererProps(400, 0, 400, 400);
	auto sink3 = boost::shared_ptr<Module>(new EglRenderer(rendererProps3));
	dummy3->setNext(sink3);


	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(dummy1);
	p->appendModule(dummy2);
	p->appendModule(dummy3);
	
	p->init();
	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	p->stop();
	p->term();
	p->wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()