#include <boost/test/unit_test.hpp>

#include "NvArgusCamera.h"
#include "NvTransform.h"
#include "FileWriterModule.h"
#include "H264EncoderV4L2.h"
#include "VirtualCameraSink.h"
#include "RTSPPusher.h"
#include "StatSink.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "DMAFDToHostCopy.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(nvarguscamera_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	auto argValue = Test_Utils::getArgValue("n", "1");
	auto n_cams = atoi(argValue.c_str());
	auto time_to_run = Test_Utils::getArgValue("s", "10");
	auto n_seconds = atoi(time_to_run.c_str());
	auto shouldrender = Test_Utils::getArgValue("r", "0");
	auto n_shouldrender = atoi(shouldrender.c_str());

	LOG_INFO << "ARGVALUE " << argValue << " n_cams " << n_cams << " n_seconds " << n_seconds << " n_shouldrender " << n_shouldrender;

	PipeLine p("test");
	for (auto i = 0; i < n_cams; i++)
	{	
		NvArgusCameraProps sourceProps(1280, 720, i);
		sourceProps.maxConcurrentFrames = 10;
		sourceProps.fps = 60;
		sourceProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
		sourceProps.qlen = 1;
		sourceProps.logHealth = true;
		sourceProps.logHealthFrequency = 100;

		auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

		if (n_shouldrender > 0)
		{
			EglRendererProps renderProps(i * 200, i * 200, 200, 200);
			renderProps.logHealth = true;
			renderProps.qlen = 1;
			renderProps.logHealthFrequency = 100;
			renderProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
			auto renderer = boost::shared_ptr<Module>(new EglRenderer(renderProps));
			source->setNext(renderer);
		}

		StatSinkProps sinkProps;
		sinkProps.logHealth = true;
		sinkProps.logHealthFrequency = 100;
		auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
		source->setNext(sink);

		p.appendModule(source);
	}
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(n_seconds));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vcam_nv12, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);
 
	VirtualCameraSinkProps sinkProps("/dev/video10");
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	copySource->setNext(sink);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12_????.raw")));
	// source->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vcam_yuv420, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	source->setNext(transform);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform->setNext(copy);

	VirtualCameraSinkProps sinkProps("/dev/video10");
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	copy->setNext(sink);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/yuv420_????.raw")));
	// copy->setNext(fileWriter);
	// auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12_????.raw")));
	// copySource->setNext(fileWriter2);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vcam, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::BGRA)));
	source->setNext(transform);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform->setNext(copy);

	VirtualCameraSinkProps sinkProps("/dev/video10");
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	copy->setNext(sink);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/bgra_????.raw")));
	// copy->setNext(fileWriter);
	// auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12_????.raw")));
	// copySource->setNext(fileWriter2);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(encoder, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(800, 800);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 4 * 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	source->setNext(encoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/ArgusCamera/frame.h264", true)));
	encoder->setNext(fileWriter);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(encoderrtsppush, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 4 * 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	source->setNext(encoder);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/ArgusCamera_1280x720.h264", true)));
	// encoder->setNext(fileWriter);

	RTSPPusherProps rtspPusherProps("rtsp://10.102.10.129:5544", "aprapipes_h264");
	auto rtspPusher = boost::shared_ptr<Module>(new RTSPPusher(rtspPusherProps));
	encoder->setNext(rtspPusher);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
