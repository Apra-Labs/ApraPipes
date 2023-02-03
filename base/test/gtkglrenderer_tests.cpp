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
// #include <constants/GlobalProperties.h>
#include <utils/GTK_UI.h>
#define PRIMARY_WINDOW_WIDTH 	1920
#define PRIMARY_WINDOW_HEIGHT 	1080

#define ASSETS_PATH "assets_ui/"
#define GLADE_PATH ASSETS_PATH "ui/"
#define STYLE_PATH ASSETS_PATH "css/"
#define CONFIG_PATH "config/"

PipeLine p("test");
GtkWidget *glarea;
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
void lauchAtlPipeline()
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 3)));

	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(transform);

	GtkGlRendererProps gtkglsinkProps(glarea, 640, 480);
	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	transform->setNext(sink);
	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));																																																															
	// transform->setNext(sink);

	p.appendModule(source);
	p.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

}
void launchPipeline()
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2)));	
	
	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(transform);

	GtkGlRendererProps gtkglsinkProps(glarea, 640, 480);
	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	transform->setNext(sink);
	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));																																																															
	// transform->setNext(sink);

	p.appendModule(source);
	p.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

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
	launchPipeline();
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

BOOST_AUTO_TEST_CASE(windowInit, *boost::unit_test::disabled())
{
	bool debug = true;
	if (debug)
	{
		printf("Hello ATL\n");
	}

	gtk_init(NULL, NULL);
	GTK_UI primaryUIUtils(GLADE_PATH "app_ui.glade", debug);
	GtkWidget *primaryScreen = primaryUIUtils.createWindow(PRIMARY_WINDOW_WIDTH,
														   PRIMARY_WINDOW_HEIGHT, true);

	
	primaryUIUtils.applyStyles(STYLE_PATH "styles.css");
	GtkWidget *fixedWnd = primaryUIUtils.getObjectFromGlade(
		"fixedLiveVidWindow");
	glarea = primaryUIUtils.getObjectFromGlade(
		"renderGlArea");
	gtk_container_add(GTK_CONTAINER(primaryScreen), fixedWnd);
	// launchPipeline();
	lauchAtlPipeline();
	gtk_widget_show_all(primaryScreen);
	gtk_main();
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
