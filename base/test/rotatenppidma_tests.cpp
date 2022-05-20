#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "NvArgusCamera.h"
#include "NvTransform.h"
#include "RotateNPPIDMA.h"
BOOST_AUTO_TEST_SUITE(rotatenppidma)

BOOST_AUTO_TEST_CASE(saveFrames)
{
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

	int width = 1280;
	int height = 720;
	boost::shared_ptr<Module> source;
	boost::shared_ptr<Module> cuctx;
    NvArgusCameraProps sourceProps(width, height, 0);
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    sourceProps.logHealth = true;
    sourceProps.logHealthFrequency = 100;
    source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));


	NvTransformProps nvtransformprops(ImageMetadata::RGBA, 1280, 720);
	nvtransformprops.qlen = 1;
	nvtransformprops.fps = 60;
	nvtransformprops.logHealth = true;
	nvtransformprops.logHealthFrequency = 100;

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(nvtransformprops));
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
    RotateNPPIDMAProps rotateprops(stream, 5.0f);
    rotateprops.qlen = 1;
    rotateprops.fps = 60;
    // rotateprops.logHealth = true;
    // rotateprops.logHealthFrequency = 100;
    auto m2 = boost::shared_ptr<RotateNPPIDMA>(new RotateNPPIDMA(rotateprops));
    nv_transform->setNext(m2);

    auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(source->init());
	BOOST_TEST(nv_transform->init());
    BOOST_TEST(m2->init());
    BOOST_TEST(m3->init());

    source->step();
    nv_transform->step();
    m2->step();
    m3->step();

	auto frames = m3->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rotatenppidma.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(camerapipeline)
{
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

	int width = 1280;
	int height = 720;
	boost::shared_ptr<Module> source;
	boost::shared_ptr<Module> cuctx;
    NvArgusCameraProps sourceProps(width, height, 0);
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    sourceProps.logHealth = true;
    sourceProps.logHealthFrequency = 100;
    source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));


	NvTransformProps nvtransformprops(ImageMetadata::RGBA, 1280, 720);
	nvtransformprops.qlen = 1;
	nvtransformprops.fps = 60;
	nvtransformprops.logHealth = true;
	nvtransformprops.logHealthFrequency = 100;

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(nvtransformprops));
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
    RotateNPPIDMAProps rotateprops(stream, 5.0f);
    rotateprops.qlen = 1;
    rotateprops.fps = 60;
    // rotateprops.logHealth = true;
    // rotateprops.logHealthFrequency = 100;
    auto m2 = boost::shared_ptr<RotateNPPIDMA>(new RotateNPPIDMA(rotateprops));
    nv_transform->setNext(m2);

    StatSinkProps sinkProps;
    // sinkProps.logHealth = true;
    // sinkProps.logHealthFrequency = 100;
    auto dummysink = boost::shared_ptr<Module>(new StatSink(sinkProps));
    m2->setNext(dummysink);
    PipeLine p("test");
	p.appendModule(source);
	p.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
	p.wait_for_all(true);
}



BOOST_AUTO_TEST_SUITE_END()