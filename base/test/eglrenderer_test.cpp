#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"


BOOST_AUTO_TEST_SUITE(eglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	constexpr int src_width = 640;
    constexpr int src_height = 360;

    for(int i = 10; i < 43; ++i)
    {
    string str = "/home/developer/ApraPipes/data/Raw_YUV420_640x360/Image0" + to_string(i) + "_YUV420.raw";
    cout<<"Processing file: " << str << endl;
    auto input_frame = makeYUV420Frame1(str, src_width, src_height);
    BOOST_REQUIRE(input_frame != nullptr);
    auto renderer = boost::make_shared<EglRenderer_test>(EglRendererProps(0, 0));
    std::string inputPinId = "input";
    framemetadata_sp metadata = input_frame->getMetadata();
    //renderer->addInputPin(metadata);

    std::string outputPinId = "output";
    frame_container frames;
    frames[inputPinId] = input_frame;

    renderer->processSOS(input_frame); 
    renderer->process(frames);
    renderer->processEOS(outputPinId);
    }
}

BOOST_AUTO_TEST_CASE(displayOnTop, *boost::unit_test::disabled())
{
	NvV4L2CameraProps nvCamProps(640,360,10,false);
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
	NvV4L2CameraProps nvCamProps(640,36,10,false);
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
	NvV4L2CameraProps nvCamProps(640,360,10,false);
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