#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "VirtualCameraSink.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "StatSink.h"
#include "H264EncoderV4L2.h"
#include "OverlayModule.h"
#include "FramesMuxer.h"
#include "EglRenderer.h"
#include "Mp4WriterSink.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(nvv4l2camera_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 10)));

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(save, *boost::unit_test::disabled())
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1920, 1080, 10)));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/frame_????.raw")));
	copySource->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vcam, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 10)));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	source->setNext(transform);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform->setNext(copy);

	auto transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::NV12)));
	source->setNext(transform2);

	auto copy2 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform2->setNext(copy2);

	// VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	// copy->setNext(sink);

	auto fileWriter1 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/uyvy_????.raw")));
	copySource->setNext(fileWriter1);

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/yuv420_????.raw")));
	copy->setNext(fileWriter2);

	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	// transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(meetup, *boost::unit_test::disabled())
{

	auto argValue = Test_Utils::getArgValue("n", "25");
	auto thresholdpipe = atoi(argValue.c_str());

	LOG_ERROR << "Using Threshold as " << thresholdpipe; 
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 2)));

	auto transform1 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	source->setNext(transform1);

	auto transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::BGRA)));
	source->setNext(transform2);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	encoderProps.enableMotionVectors = true;
	encoderProps.motionVectorThreshold = thresholdpipe;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	transform1->setNext(encoder);

	std::vector<std::string> encodedImagePin;
	encodedImagePin = encoder->getAllOutputPinsByType(FrameMetadata::H264_DATA);

	auto copySource2 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform2->setNext(copySource2);

	auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
	encoder->setNext(muxer);
	copySource2->setNext(muxer);

	auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	muxer->setNext(overlay);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	overlay->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();

}
BOOST_AUTO_TEST_SUITE_END()
