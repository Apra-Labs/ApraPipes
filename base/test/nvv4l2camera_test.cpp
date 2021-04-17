#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "VirtualCameraSink.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "StatSink.h"
#include "EglRenderer.h"

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

BOOST_AUTO_TEST_SUITE_END()
