#include <boost/test/unit_test.hpp>
#include <memory>
#include <thread>
#include <chrono>

#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"


BOOST_AUTO_TEST_SUITE(eglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	int width = 640;
	int height = 480;

    FileReaderModuleProps fileReaderProps("./data/ArgusCamera");
	fileReaderProps.fps = 30;
	auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC1, 0, CV_8U, FrameMetadata::MemType::DMABUF, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto sink = std::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(displayOnTop, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = std::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = std::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = std::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(switch_display, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = std::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = std::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = std::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(open_close_window, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = std::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = std::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = std::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(5));
	sink->closeWindow();

	std::this_thread::sleep_for(std::chrono::seconds(10));
	sink->createWindow(200,200);

	std::this_thread::sleep_for(std::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_SUITE_END()