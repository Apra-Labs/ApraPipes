#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "VirtualCameraSink.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "StatSink.h"
#include "EglRenderer.h"
#include "GtkGlRenderer.h"
#include "NvArgusCamera.h"
#include <gtk/gtk.h>
#include "ValveModule.h"
#include "Mp4WriterSink.h"
#include "EndocamControlModule.h"
#include "H264EncoderV4L2.h"
// #include <utils/GTK_UI.h>
#include "RotateNPPI.h"
#define PRIMARY_WINDOW_WIDTH 1920
#define PRIMARY_WINDOW_HEIGHT 1080

#define ASSETS_PATH "assets_ui/"
#define GLADE_PATH ASSETS_PATH "ui/"
#define STYLE_PATH ASSETS_PATH "css/"
#define CONFIG_PATH "config/"
// #define STYLE_PATH ASSETS_PATH "/css"

bool isRecording = false;
boost::shared_ptr<ValveModule> valve;
boost::shared_ptr<ValveModule> recordvalve;
boost::shared_ptr<Mp4WriterSink> mp4WriterSink;
boost::shared_ptr<RotateNPPI> rotateMod;
PipeLine p("test");
GtkWidget *glarea, *mp4Record, *captureBtn, *rotate90;
BOOST_AUTO_TEST_SUITE(gtkglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{

	// Logger::setLogLevel(boost::log::trivial::severity_level::info);

	// auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2)));

	// GtkGlRendererProps gtkglsinkProps("atlui.glade", 1920, 1080);

	// auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	// source->setNext(sink);

	// PipeLine p("test");
	// p.appendModule(source);
	// BOOST_TEST(p.init());

	// p.run_all_threaded();
	// boost::this_thread::sleep_for(boost::chrono::seconds(10000000000));
	// gtk_main();
	// p.stop();
	// p.term();
	// p.wait_for_all();
}

static gboolean startRecording(GtkWidget *widget, GdkEvent *event, gpointer data)
{
	LOG_ERROR << "start Recording Button Pressed";
	if (isRecording)
	{
		recordvalve->allowFrames(0);
		mp4WriterSink->closeFile();
	}
	else
	{
		recordvalve->allowFrames(-1);

	}
	isRecording = !isRecording;
	return TRUE;
}

static gboolean captureFrame(GtkWidget *widget, GdkEvent *event, gpointer data)
{
	LOG_ERROR << "Cature Frame is Clicked";
	valve->allowFrames(2);
	return TRUE;
}

static gboolean rotateCam(GtkWidget *widget, GdkEvent *event, gpointer data)
{
	LOG_ERROR << "rotate Cam is Clicked";
    auto currRotateProps = rotateMod->getProps();
	currRotateProps.angle = fmod((currRotateProps.angle + 90.0), 360.0);
	rotateMod->setProps(currRotateProps);
	return TRUE;
}

void lauchAtlPipeline()
{
	// Logger::setLogLevel(boost::log::trivial::severity_level::info);
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/xyz.mp4";
	// auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2)));
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 3)));

	auto nvtransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nvtransform);

    // double angle = 0.0f;
	// auto stream = cudastream_sp(new ApraCudaStream);
	// rotateMod = boost::shared_ptr<RotateNPPI>(new RotateNPPI(RotateNPPIProps(stream, angle)));
	// nvtransform->setNext(rotateMod);

	GtkGlRendererProps gtkglsinkProps(glarea, 640, 480);
	auto sink = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
	nvtransform->setNext(sink);

	// frame Capture
	valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(1)));
	nvtransform->setNext(valve);

	auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	valve->setNext(dmaToHostCopy);

	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/OutputImages/Frame???.raw")));
	dmaToHostCopy->setNext(fileWriter);

	// close

	// mp4 write
	recordvalve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(0)));
	nvtransform->setNext(recordvalve);

	auto nvtransformtoy20 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	recordvalve->setNext(nvtransformtoy20);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	nvtransformtoy20->setNext(encoder);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(UINT32_MAX, 10, 24, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	encoder->setNext(mp4WriterSink);

	// close mp4 write

	auto mControl = boost::shared_ptr<EndocamControlModule>(new EndocamControlModule(EndocamControlModuleProps()));
	// Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.appendModule(source);
	p.addControlModule(mControl);
	mControl->enrollModule("valve", valve);

	p.init();
	mControl->init();
	p.run_all_threaded();
}
void launchRendererPipelineUSB()
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2)));

	auto nvtransform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nvtransform);

	GtkGlRendererProps gtkglsinkProps(glarea, 640, 480);
	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	nvtransform->setNext(sink);

	auto mControl = boost::shared_ptr<EndocamControlModule>(new EndocamControlModule(EndocamControlModuleProps()));
	p.appendModule(source);
	p.addControlModule(mControl);
	// mControl->enrollModule("valve", valve);
	p.init();

	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
	// boost::this_thread::sleep_for(boost::chrono::seconds(100));
}

void launchPipeline()
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	std::string outFolderPath = "./data/testOutput/mp4_videos/rgb_24bpp/xyz.mp4";

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 10)));

	auto nvtransform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nvtransform);

	GtkGlRendererProps gtkglsinkProps(glarea, 640, 480);
	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	nvtransform->setNext(sink);

	/// block For Saving Screenshot File
	// valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(1)));
	// nvtransform->setNext(valve);

	// auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// valve->setNext(dmaToHostCopy);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/OutputImages/Frame???.raw")));
	// dmaToHostCopy->setNext(fileWriter);

	// /// block For Saving recording
	// recordvalve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(0)));
	// source->setNext(recordvalve);

	// auto nvtransformtoy20 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	// recordvalve->setNext(nvtransformtoy20);

	// H264EncoderV4L2Props encoderProps;
	// encoderProps.targetKbps = 1024;
	// auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	// nvtransformtoy20->setNext(encoder);

	// auto mp4WriterSinkProps = Mp4WriterSinkProps(UINT32_MAX, 10, 24, outFolderPath);
	// mp4WriterSinkProps.logHealth = true;
	// mp4WriterSinkProps.logHealthFrequency = 100;
	// mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	// encoder->setNext(mp4WriterSink);

	// // auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));
	// // transform->setNext(sink);

	// auto mControl = boost::shared_ptr<EndocamControlModule>(new EndocamControlModule(EndocamControlModuleProps()));
	p.appendModule(source);
	//p.addControlModule(mControl);
	//mControl->enrollModule("valve", valve);
	p.init();

	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(400));
}
void screenChanged(GtkWidget *widget, GdkScreen *old_screen,
				   gpointer userdata)
{
	/* To check if the display supports alpha channels, get the visual */
	GdkScreen *screen = gtk_widget_get_screen(widget);
	GdkVisual *visual = gdk_screen_get_rgba_visual(screen);
	if (!visual)
	{
		printf("Your screen does not support alpha channels!\n");
		visual = gdk_screen_get_system_visual(screen);
	}
	else
	{
		printf("Your screen supports alpha channels!\n");
	}
	gtk_widget_set_visual(widget, visual);
}

BOOST_AUTO_TEST_CASE(windowInit2, *boost::unit_test::disabled())
{
	if (!gtk_init_check(NULL, NULL)) // yash argc argv
	{
		fputs("Could not initialize GTK", stderr);
	}
	GtkBuilder *m_builder = gtk_builder_new();
	if (!m_builder)
	{
		LOG_ERROR << "Builder not found";
	}
	gtk_builder_add_from_file(m_builder, "atlui.glade", NULL);

	GtkWidget *window = GTK_WIDGET(gtk_window_new(GTK_WINDOW_TOPLEVEL));
	g_object_ref(window);
	gtk_window_set_default_size(GTK_WINDOW(window), 1920, 1080);
	gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
	gtk_widget_set_app_paintable(window, TRUE);

	do
	{
		gtk_main_iteration();
	} while (gtk_events_pending());

	GtkWidget *mainFixed = GTK_WIDGET(gtk_builder_get_object(m_builder, "mainWidget"));
	gtk_container_add(GTK_CONTAINER(window), mainFixed);
	glarea = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw"));
	g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

	// do
	// {
	// 	gtk_main_iteration();
	// } while (gtk_events_pending());
	// launchPipeline();
	// launchRendererPipelineUSB();
	// lauchAtlPipeline();
	// std::thread pipelineThread();
	// std::thread &&threaddbar = std::thread(launchPipeline);
	// pipelineThread = std::move(threaddbar);
	gtk_widget_show_all(window);
	gtk_main();

	p.stop();
	p.term();
	p.wait_for_all();
	// pipelineThread.join();
}

// BOOST_AUTO_TEST_CASE(windowInit, *boost::unit_test::disabled())
// {
// 	bool debug = false;
// 	if (debug)
// 	{
// 		printf("Hello ATL\n");
// 	}
// 	std::string css_path = "/css/styles.css";
// 	gtk_init(NULL, NULL);
// 	apra::GTK_UI primaryUIUtils(GLADE_PATH "app_ui.glade", debug);
// 	GtkWidget *primaryScreen = primaryUIUtils.createWindow((uint)PRIMARY_WINDOW_WIDTH,
// 														   (uint)PRIMARY_WINDOW_HEIGHT, true, true);

// 	primaryUIUtils.applyStyles(STYLE_PATH "styles.css");
// 	GtkWidget *fixedWnd = primaryUIUtils.getObjectFromGlade(
// 		"fixedLiveVidWindow");
// 	glarea = primaryUIUtils.getObjectFromGlade(
// 		"renderGlArea");
// 	gtk_container_add(GTK_CONTAINER(primaryScreen), fixedWnd);
// 	// launchPipeline();
// 	lauchAtlPipeline();
// 	gtk_widget_show_all(primaryScreen);
// 	gtk_main();
// 	p.stop();
// 	p.term();
// 	p.wait_for_all();
// }

// BOOST_AUTO_TEST_CASE(Application, *boost::unit_test::disabled())
// {
// 	bool debug = false;
// 	if (debug)
// 	{
// 		printf("Hello ATL\n");
// 	}
// 	std::string css_path = "/css/styles.css";
// 	gtk_init(NULL, NULL);
// 	apra::GTK_UI primaryUIUtils(GLADE_PATH "app_ui.glade", debug);
// 	GtkWidget *primaryScreen = primaryUIUtils.createWindow((uint)PRIMARY_WINDOW_WIDTH,
// 														   (uint)PRIMARY_WINDOW_HEIGHT, true, true);

// 	primaryUIUtils.applyStyles(STYLE_PATH "styles.css");
// 	GtkWidget *fixedWnd = primaryUIUtils.getObjectFromGlade(
// 		"fixedLiveVidWindow");
// 	glarea = primaryUIUtils.getObjectFromGlade(
// 		"renderGlArea");
// 	mp4Record = primaryUIUtils.getObjectFromGlade(
// 		"recordBtn");
// 	captureBtn = primaryUIUtils.getObjectFromGlade(
// 		"captureBtn");
// 	rotate90 = primaryUIUtils.getObjectFromGlade(
// 		"rotateVidBtn");

// 	g_signal_connect(G_OBJECT(mp4Record), "button-press-event", G_CALLBACK(startRecording), NULL);
// 	g_signal_connect(G_OBJECT(captureBtn), "button-press-event", G_CALLBACK(captureFrame), NULL);
// 	g_signal_connect(G_OBJECT(rotate90), "button-press-event", G_CALLBACK(rotateCam), NULL);

// 	g_signal_connect(G_OBJECT(mp4Record), "clicked", G_CALLBACK(startRecording), NULL);
// 	g_signal_connect(G_OBJECT(captureBtn), "clicked", G_CALLBACK(captureFrame), NULL);
// 	g_signal_connect(G_OBJECT(rotate90), "clicked", G_CALLBACK(rotateCam), NULL);

// 	gtk_container_add(GTK_CONTAINER(primaryScreen), fixedWnd);
// 	// launchPipeline();
// 	launchRendererPipelineUSB();
// 	gtk_widget_show_all(primaryScreen);
// 	gtk_main();
// 	p.stop();
// 	p.term();
// 	p.wait_for_all();
// }

BOOST_AUTO_TEST_SUITE_END()
