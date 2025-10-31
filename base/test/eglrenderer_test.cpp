#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"


BOOST_AUTO_TEST_SUITE(eglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	int width = 640;
	int height = 360;

    FileReaderModuleProps fileReaderProps("/home/developer/ApraPipes/data/Raw_YUV420_640x360/Image001_YUV420.raw");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC1, 0, CV_8U, FrameMetadata::MemType::DMABUF, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(displayOnTop, *boost::unit_test::disabled())
{
	LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
	NvV4L2CameraProps nvCamProps(640,360, 10,false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(switch_display, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10,false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(open_close_window, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10,false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	sink->closeWindow();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	sink->createWindow(200,200);

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(displayOnTop, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(switch_display, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(open_close_window, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360, 10);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	sink->closeWindow();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	sink->createWindow(200,200);

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all(); 
}

BOOST_AUTO_TEST_SUITE_END()