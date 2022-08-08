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
#include "V4L2CameraSource.h"
#include "HostDMA.h"
#include "BayerToRGBA.h"
#include "CudaMemCopy.h"
#include "DeviceToDMA.h"
#include "ImageResizeCV.h"
#include "ResizeNPPI.h"
#include "BayerToGray.h"
#include "RestrictCapFrames.h"
#include "DeviceToDMAMono.h"
#include "BayerToMono.h"
BOOST_AUTO_TEST_SUITE(nvarguscamera_tests)

BOOST_AUTO_TEST_CASE(basiccolorchange, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720, 0);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	sourceProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
	sourceProps.qlen = 1;
	// sourceProps.logHealth = true;
	// sourceProps.logHealthFrequency = 100;

	auto source = boost::shared_ptr<NvArgusCamera>(new NvArgusCamera(sourceProps));

	EglRendererProps renderProps(1 * 200, 1 * 200, 1 * 200, 1 * 200);
	renderProps.logHealth = true;
	renderProps.qlen = 1;
	renderProps.logHealthFrequency = 100;
	renderProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
	auto renderer = boost::shared_ptr<Module>(new EglRenderer(renderProps));
	source->setNext(renderer);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(15));
	LOG_ERROR << "Closing  Pipeline";
	source->enableAutoWB();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	source->disableAutoWB();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));

	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	auto argValue = Test_Utils::getArgValue("n", "3");
	auto n_cams = atoi(argValue.c_str());
	auto time_to_run = Test_Utils::getArgValue("s", "100000");
	auto n_seconds = atoi(time_to_run.c_str());
	auto shouldrender = Test_Utils::getArgValue("r", "1");
	auto n_shouldrender = atoi(shouldrender.c_str());

	LOG_ERROR << "ARGVALUE " << argValue << " n_cams " << n_cams << " n_seconds " << n_seconds << " n_shouldrender " << n_shouldrender;

	PipeLine p("test");
	for (auto i = 0; i < n_cams; i++)
	{
		NvArgusCameraProps sourceProps(1280, 720, i);
		sourceProps.maxConcurrentFrames = 10;
		sourceProps.fps = 60;
		sourceProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
		sourceProps.qlen = 1;
		// sourceProps.logHealth = true;
		// sourceProps.logHealthFrequency = 100;

		auto source = boost::shared_ptr<NvArgusCamera>(new NvArgusCamera(sourceProps));

		if (n_shouldrender > 0)
		{
			EglRendererProps renderProps(i * 200, i * 200, i * 200, i * 200);
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
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	LOG_ERROR << "Closing  Pipeline";
	// // source->enableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));
	// LOG_ERROR << "Pipeline COmplete";
	// source->disableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));
	// source->enableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->disableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->enableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->disableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->enableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->disableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	// source->enableAutoWB();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10));

	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	LOG_ERROR << "Stop Done";
	p.term();
	LOG_ERROR << "Terminate Done";
	p.wait_for_all();
	LOG_ERROR << "Wait for All Done";
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

// BOOST_AUTO_TEST_CASE(vcam_yuv420, *boost::unit_test::disabled())
// {
// 	NvArgusCameraProps sourceProps(1280, 720);
// 	sourceProps.maxConcurrentFrames = 10;
// 	sourceProps.fps = 30;
// 	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

// 	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
// 	source->setNext(copySource);

// 	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
// 	source->setNext(transform);

// 	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
// 	transform->setNext(copy);

// 	VirtualCameraSinkProps sinkProps("/dev/video10");
// 	sinkProps.logHealth = true;
// 	sinkProps.logHealthFrequency = 100;
// 	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
// 	copy->setNext(sink);

// 	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/yuv420_????.raw")));
// 	// copy->setNext(fileWriter);
// 	// auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12_????.raw")));
// 	// copySource->setNext(fileWriter2);

// 	PipeLine p("test");
// 	p.appendModule(source);
// 	BOOST_TEST(p.init());

// 	Logger::setLogLevel(boost::log::trivial::severity_level::info);

// 	p.run_all_threaded();

// 	boost::this_thread::sleep_for(boost::chrono::seconds(100));
// 	Logger::setLogLevel(boost::log::trivial::severity_level::error);

// 	p.stop();
// 	p.term();

// 	p.wait_for_all();
// }

// BOOST_AUTO_TEST_CASE(vcam, *boost::unit_test::disabled())
// {
// 	NvArgusCameraProps sourceProps(1280, 720);
// 	sourceProps.maxConcurrentFrames = 10;
// 	sourceProps.fps = 30;
// 	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

// 	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
// 	source->setNext(copySource);

// 	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::BGRA)));
// 	source->setNext(transform);

// 	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
// 	transform->setNext(copy);

// 	VirtualCameraSinkProps sinkProps("/dev/video10");
// 	sinkProps.logHealth = true;
// 	sinkProps.logHealthFrequency = 100;
// 	auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
// 	copy->setNext(sink);

// 	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/bgra_????.raw")));
// 	// copy->setNext(fileWriter);
// 	// auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12_????.raw")));
// 	// copySource->setNext(fileWriter2);

// 	PipeLine p("test");
// 	p.appendModule(source);
// 	BOOST_TEST(p.init());

// 	Logger::setLogLevel(boost::log::trivial::severity_level::info);

// 	p.run_all_threaded();

// 	boost::this_thread::sleep_for(boost::chrono::seconds(100));
// 	Logger::setLogLevel(boost::log::trivial::severity_level::error);

// 	p.stop();
// 	p.term();

// 	p.wait_for_all();
// }

BOOST_AUTO_TEST_CASE(encoder, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps(1280, 720);
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


BOOST_AUTO_TEST_CASE(v4l2cam, *boost::unit_test::disabled())   /// working
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(BayerToRGBAProps()));
	source->setNext(bayerTRGBA);

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);


	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTRGBA->setNext(hostdma);

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::NV12, 100)));
	hostdma->setNext(nv_transform);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());


	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camuyvy, *boost::unit_test::disabled())   /// working
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 10;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	source->setNext(hostdma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	hostdma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camnotrans, *boost::unit_test::disabled()) //working
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(BayerToRGBAProps()));
	source->setNext(bayerTRGBA);

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTRGBA->setNext(hostdma);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	hostdma->setNext(copy);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameA????.raw", true)));
	copy->setNext(fileWriter);

	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	// hostdma->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camrend, *boost::unit_test::disabled()) // notworking
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(BayerToRGBAProps()));
	source->setNext(bayerTRGBA);

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTRGBA->setNext(hostdma);

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::BGRA, 100)));
	hostdma->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	// Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100000));
	// Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(mpcdy, *boost::unit_test::disabled()) // notworking
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(BayerToRGBAProps()));
	source->setNext(bayerTRGBA);

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTRGBA->setNext(hostdma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 1024, 800)));
	hostdma->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	// Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100000));
	// Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camplzworkwithoutrender, *boost::unit_test::disabled()) //working
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTGray = boost::shared_ptr<Module>(new BayerToGray(BayerToGrayProps()));
	source->setNext(bayerTGray);

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTGray->setNext(hostdma);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	hostdma->setNext(copy);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameA????.raw", true)));
	copy->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camplzworkwithrender, *boost::unit_test::disabled()) //working
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTGray = boost::shared_ptr<Module>(new BayerToGray(BayerToGrayProps()));
	source->setNext(bayerTGray);

	HostDMAProps hostdmaprops;
	hostdmaprops.qlen = 1;
	hostdmaprops.logHealth = true;
	hostdmaprops.logHealthFrequency = 100;

	auto hostdma = boost::shared_ptr<Module>(new HostDMA(hostdmaprops));
	bayerTGray->setNext(hostdma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 1024, 800)));
	hostdma->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camsave, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0"); /// working
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/aav4l2frame?????.raw", false, false)));
	source->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(v4l2camsavexternal, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0"); /// working
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/SingleCam/frame?????.raw", false, false)));
	source->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(v4l2camsave2, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(BayerToRGBAProps()));
	source->setNext(bayerTRGBA);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/newframe?????.raw", true)));
	bayerTRGBA->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(v4l2camdevice, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	// auto resize = boost::shared_ptr<ImageResizeCV>(new ImageResizeCV(ImageResizeCVProps(400,400)));
	// source->setNext(resize);

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops)); //changing dim revback
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	bayerTRGBA->setNext(copy);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	copy->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedma->setNext(sink);

	// StatSinkProps sinkProps;
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	// devicedma->setNext(sink);

	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	// hostdma->setNext(sink);
	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/newO/frameA????.raw", true)));
	// bayerTRGBA->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(v4l2camdeviceresize, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	bayerTRGBA->setNext(copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(400, 400, stream)));
	copy->setNext(resize);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	resize->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 400, 400)));
	devicedma->setNext(sink);

	// StatSinkProps sinkProps;
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	// devicedma->setNext(sink);

	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	// hostdma->setNext(sink);
	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/newO/frameA????.raw", true)));
	// bayerTRGBA->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(v4l2camdeviceresizenorender, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	bayerTRGBA->setNext(copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(400, 400, stream)));
	copy->setNext(resize);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	resize->setNext(sink);


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



BOOST_AUTO_TEST_CASE(v4l2camdeviceresize960, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	bayerTRGBA->setNext(copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 960, stream)));
	copy->setNext(resize);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	resize->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 960, 960)));
	devicedma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camdevicenooper, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	bayerTRGBA->setNext(copy);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	copy->setNext(sink);


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
BOOST_AUTO_TEST_CASE(v4l2camNoRender, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	devicedma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camRender, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camRenderWrite, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput/frameA????.raw", false, false)));
	source->setNext(fileWriter);

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedma->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camRenderWrite2Cam, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput/frameA????.raw", false, false)));
	source->setNext(fileWriter);

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedma->setNext(sink);

	V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 60;
	auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput2/frameA????.raw", false, false)));
	source2->setNext(fileWriter2);

	BayerToRGBAProps brgbprops2;
	brgbprops2.qlen = 1;
	auto bayerTRGBA2 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops2));
	source2->setNext(bayerTRGBA2);

	DeviceToDMAProps deviceTodmaprops2(stream);
	deviceTodmaprops2.qlen = 1;
	deviceTodmaprops2.logHealth = true;
	deviceTodmaprops2.logHealthFrequency = 100;

	auto devicedma2 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops2));
	bayerTRGBA2->setNext(devicedma2);

	auto sink2 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 800, 800, 800)));
	devicedma2->setNext(sink2);

	PipeLine p("test");
	p.appendModule(source);
	p.appendModule(source2);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camRenderWrite3Cam, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput/frameA????.raw", false, false)));
	source->setNext(fileWriter);

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedma->setNext(sink);

	V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 60;
	auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput2/frameA????.raw", false, false)));
	source2->setNext(fileWriter2);

	BayerToRGBAProps brgbprops2;
	brgbprops2.qlen = 1;
	auto bayerTRGBA2 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops2));
	source2->setNext(bayerTRGBA2);

	DeviceToDMAProps deviceTodmaprops2(stream);
	deviceTodmaprops2.qlen = 1;
	deviceTodmaprops2.logHealth = true;
	deviceTodmaprops2.logHealthFrequency = 100;

	auto devicedma2 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops2));
	bayerTRGBA2->setNext(devicedma2);

	auto sink2 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 800, 800, 800)));
	devicedma2->setNext(sink2);

	V4L2CameraSourceProps sourceProps3(800, 800, "/dev/video2");
	sourceProps3.maxConcurrentFrames = 10;
	sourceProps3.fps = 60;
	auto source3 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps3));

	auto fileWriter3 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput3/frameA????.raw", false, false)));
	source3->setNext(fileWriter3);

	BayerToRGBAProps brgbprops3;
	brgbprops3.qlen = 1;
	auto bayerTRGBA3 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops3));
	source3->setNext(bayerTRGBA3);

	DeviceToDMAProps deviceTodmaprops3(stream);
	deviceTodmaprops3.qlen = 1;
	deviceTodmaprops3.logHealth = true;
	deviceTodmaprops3.logHealthFrequency = 100;

	auto devicedma3 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops3));
	bayerTRGBA3->setNext(devicedma3);

	auto sink3 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(800, 0, 800, 800)));
	devicedma3->setNext(sink3);

	PipeLine p("test");
	p.appendModule(source);
	p.appendModule(source2);
	p.appendModule(source3);
	
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(v4l2camRendercsi, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(640, 480, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 640, 360)));
	devicedma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camRendermono, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToMonoProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTMono = boost::shared_ptr<Module>(new BayerToMono(brgbprops));
	source->setNext(bayerTMono);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAMonoProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedmamono = boost::shared_ptr<Module>(new DeviceToDMAMono(deviceTodmaprops));
	bayerTMono->setNext(devicedmamono);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
	devicedmamono->setNext(sink);

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

BOOST_AUTO_TEST_CASE(camerawrite, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	BayerToRGBAProps brgbprops;
	brgbprops.qlen = 1;
	auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
	source->setNext(bayerTRGBA);

	auto stream = cudastream_sp(new ApraCudaStream);

	DeviceToDMAProps deviceTodmaprops(stream);
	deviceTodmaprops.qlen = 1;
	deviceTodmaprops.logHealth = true;
	deviceTodmaprops.logHealthFrequency = 100;

	auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
	bayerTRGBA->setNext(devicedma);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	devicedma->setNext(copy);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameA????.raw", false, true)));
	source->setNext(fileWriter);

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

BOOST_AUTO_TEST_CASE(camerawriteAll, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 60;
	auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

	V4L2CameraSourceProps sourceProps3(800, 800, "/dev/video2");
	sourceProps3.maxConcurrentFrames = 10;
	sourceProps3.fps = 60;
	auto source3 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps3));


	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameA????.raw", false, false)));
	source->setNext(fileWriter);

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameB????.raw", false, false)));
	source2->setNext(fileWriter2);


	auto fileWriter3 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameC????.raw", false, false)));
	source3->setNext(fileWriter3);


	PipeLine p("test");
	p.appendModule(source);
	p.appendModule(source2);
	p.appendModule(source3);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(argusWriteAll, *boost::unit_test::disabled())
{
	NvArgusCameraProps sourceProps1(800, 800, 0);
	sourceProps1.maxConcurrentFrames = 10;
	sourceProps1.fps = 30;
	auto source1 = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps1));

	auto copySource1 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source1->setNext(copySource1);

	NvArgusCameraProps sourceProps2(800, 800, 1);
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 30;
	auto source2 = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps2));

	auto copySource2 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source2->setNext(copySource2);

	NvArgusCameraProps sourceProps3(800, 800, 2);
	sourceProps3.maxConcurrentFrames = 10;
	sourceProps3.fps = 30;
	auto source3 = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps3));

	auto copySource3 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source3->setNext(copySource3);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/Argus/frameA????.raw", false, false)));
	copySource1->setNext(fileWriter);

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/Argus/frameB????.raw", false, false)));
	copySource2->setNext(fileWriter2);

	auto fileWriter3 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/Argus/frameC????.raw", false, false)));
	copySource3->setNext(fileWriter3);

	PipeLine p("test");
	p.appendModule(source1);
	p.appendModule(source2);
	p.appendModule(source3);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();
	LOG_ERROR << "Sleeping for 10 seconds";
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	
	LOG_ERROR << "Please Wait we are closing the pipeline";
	p.stop();
	LOG_ERROR << "Pipelone stop call is finished";
	p.term();
	LOG_ERROR << "Pipelone Terminate call is finished";
	p.wait_for_all();
	LOG_ERROR << "Pipelone Wait for all is finished";
}

BOOST_AUTO_TEST_CASE(cameraSinkAll, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	// Logger::setLogLevel(boost::log::trivial::severity_level::error);

	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 60;
	auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

	V4L2CameraSourceProps sourceProps3(800, 800, "/dev/video2");
	sourceProps3.maxConcurrentFrames = 10;
	sourceProps3.fps = 60;
	auto source3 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps3));

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink1 = boost::shared_ptr<Module>(new StatSink(sinkProps));

	StatSinkProps sinkProps2;
	sinkProps2.logHealth = true;
	sinkProps2.logHealthFrequency = 100;
	auto sink2 = boost::shared_ptr<Module>(new StatSink(sinkProps2));

	StatSinkProps sinkProps3;
	sinkProps3.logHealth = true;
	sinkProps3.logHealthFrequency = 100;
	auto sink3 = boost::shared_ptr<Module>(new StatSink(sinkProps3));

	source->setNext(sink1);
	source2->setNext(sink2);
	source3->setNext(sink3);
	PipeLine p("test");
	p.appendModule(source);
	p.appendModule(source2);
	p.appendModule(source3);
	BOOST_TEST(p.init());

	
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	
	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(camerawriteAllExternal, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

	V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
	sourceProps2.maxConcurrentFrames = 10;
	sourceProps2.fps = 60;
	auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

	V4L2CameraSourceProps sourceProps3(800, 800, "/dev/video2");
	sourceProps3.maxConcurrentFrames = 10;
	sourceProps3.fps = 60;
	auto source3 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps3));


	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameA????.raw", false, false)));
	source->setNext(fileWriter);

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameB????.raw", false, false)));
	source2->setNext(fileWriter2);


	auto fileWriter3 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameC????.raw", false, false)));
	source3->setNext(fileWriter3);


	PipeLine p("test");
	p.appendModule(source);
	p.appendModule(source2);
	p.appendModule(source3);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}


BOOST_AUTO_TEST_CASE(v4l2camsaveqq, *boost::unit_test::disabled())
{
	V4L2CameraSourceProps sourceProps1(800, 800, "/dev/video0");
	sourceProps1.maxConcurrentFrames = 10;
	sourceProps1.fps = 60;
	auto source1 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps1));

	RestrictCapFramesProps restcam(10);
	auto restCapFrames = boost::shared_ptr<RestrictCapFrames>(new RestrictCapFrames(restcam));
	source1->setNext(restCapFrames);

	auto filewriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./V4L2Output/ImageA?????.raw", true, false)));
	restCapFrames->setNext(filewriter);

	PipeLine p("test");
	p.appendModule(source1);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	restCapFrames->resetFrameCapture();
	boost::this_thread::sleep_for(boost::chrono::seconds(2000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);
	p.stop();
	p.term();

	p.wait_for_all();
}


BOOST_AUTO_TEST_SUITE_END()
