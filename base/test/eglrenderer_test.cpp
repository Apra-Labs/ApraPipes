#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "NvEglRenderer.h"
#include "ApraNvEglRenderer.h"

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
	//,"/home/developer/ApraPipes/data/Debrosee-ALPnL.ttf","HelloWorld",1.0f,1.0f,1.0f,1.0f,24,200,200,0.99)));
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
	//,"/home/developer/ApraPipes/data/Debrosee-ALPnL.ttf","HelloWorld",1.0f,1.0f,1.0f,1.0f,24,10,50)));
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

BOOST_AUTO_TEST_CASE(ctor_default, *boost::unit_test::disabled())
{
    LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps()));
    transform->setNext(sink);

    PipeLine p("default_ctor_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_xy, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(50, 50)));
    transform->setNext(sink);

    PipeLine p("geometry_xy_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_xywh, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 320, 240)));
    transform->setNext(sink);

    PipeLine p("geometry_xywh_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_text, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    EglRendererProps::TextInfo text;
    text.fontPath = "/home/developer/ApraPipes/data/Debrosee-ALPnL.ttf";
    text.message = "HelloText";
    text.color = {1.0f, 0.0f, 1.0f};
    text.fontSize = 24;
    text.position = {50,50};
	text.scale = 1;
    text.opacity = 1.0f;
    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(100, 100, 320, 240, text)));
    transform->setNext(sink);

    PipeLine p("geometry_text_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_image, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    EglRendererProps::ImageInfo img;
    img.path = "/home/developer/ApraPipes/data/apra.jpeg";
    img.position = {0, 0};
    img.size = {128, 128};
    img.opacity = 0.75f;
    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(200, 200, 320, 240, img)));
    transform->setNext(sink);

    PipeLine p("geometry_image_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_text_image, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    EglRendererProps::TextInfo text;
    text.fontPath = "/home/developer/ApraPipes/data/Debrosee-ALPnL.ttf";
    text.message = "Overlay Text";
    text.color = {0.0f, 1.0f, 0.0f};
    text.fontSize = 20;
    text.position = {100, 50};

    EglRendererProps::ImageInfo img;
    img.path = "/home/developer/ApraPipes/data/apra.jpeg";
    img.position = {200, 150};
    img.size = {64, 64};

    auto sink = boost::shared_ptr<Module>(
        new EglRenderer(EglRendererProps(0, 0, 320, 240, text, img))
    );
    transform->setNext(sink);

    PipeLine p("geometry_text_image_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(ctor_geometry_opacity_mask, *boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::trace);

    NvV4L2CameraProps camProps(640, 360, 10, false);
    auto source = boost::shared_ptr<Module>(new NvV4L2Camera(camProps));

    NvTransformProps nvprops(ImageMetadata::RGBA);
    auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
    source->setNext(transform);

    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 320, 240, 0.5f, true)));
    transform->setNext(sink);

    PipeLine p("geometry_opacity_mask_test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop(); p.term(); p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()