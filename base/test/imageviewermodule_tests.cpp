#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "PipeLine.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "ImageViewerModule.h"
#include "WebCamSource.h"

#if defined(__arm__) || defined(__aarch64__)
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#endif

BOOST_AUTO_TEST_SUITE(imageviewermodule_tests)

BOOST_AUTO_TEST_CASE(Dma_Renderer_Planarimage, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::NV12));
	source->setNext(transform);

	auto sink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps(0, 0, 0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(5));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_CASE(Dma_Renderer_Rawimage, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::RGBA));
	source->setNext(transform);

	auto sink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps(0, 0, 1)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(40));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_CASE(open_close_window, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto sink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps(0, 0, 0)));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::seconds(5));
	sink->closeWindow();

	std::this_thread::sleep_for(std::chrono::seconds(10));
	sink->createWindow(200, 200);

	std::this_thread::sleep_for(std::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_CASE(viewer_test, *boost::unit_test::disabled())
{
	WebCamSourceProps webCamSourceprops(-1, 640, 480);
	auto source = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	std::this_thread::sleep_for(std::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
