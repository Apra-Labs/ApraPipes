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

BOOST_AUTO_TEST_SUITE(gtkglrenderer_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2)));

	GtkGlRendererProps gtkglsinkProps("atlui.glade", 1920, 1080);

	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10000000000));
	gtk_main();
	p.stop();
	p.term();
	p.wait_for_all();
}

void launchPipeline ()
{
	NvArgusCameraProps sourceProps(1280, 720, 0);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	EglRendererProps renderProps(200, 200, 1280, 720);
	renderProps.logHealth = true;
	renderProps.qlen = 1;
	renderProps.logHealthFrequency = 100;
	auto renderer = boost::shared_ptr<Module>(new EglRenderer(renderProps));
	source->setNext(renderer);

	PipeLine p("test");
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

}

BOOST_AUTO_TEST_CASE(windowInit, *boost::unit_test::disabled())
{
	if (!gtk_init_check(NULL, NULL)) // yash argc argv
	{
		fputs("Could not initialize GTK", stderr);
		
	}

	// Create toplevel window, add GtkGLArea:
	GtkBuilder *m_builder = gtk_builder_new();
	if (!m_builder)
	{
		LOG_ERROR << "Builder not found";
		
	}
	gtk_builder_add_from_file(m_builder, "atlui.glade", NULL);

	GtkWidget *window = (GtkWidget *)gtk_window_new(GTK_WINDOW_TOPLEVEL);
	g_object_ref(window);
	gtk_window_set_default_size(GTK_WINDOW(window), 1920, 1080);
	gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
	gtk_widget_set_app_paintable(window, TRUE);

	// screenChanged(window, NULL, NULL);
	do
	{
		gtk_main_iteration();
	} while (gtk_events_pending());

	GtkWidget *mainFixed = (GtkWidget *)gtk_builder_get_object(m_builder, "mainWidget");
	gtk_container_add(GTK_CONTAINER(window), mainFixed);
	GtkWidget *glarea = (GtkWidget *)gtk_builder_get_object(m_builder, "glareadraw");

	// Connect GTK signals:
	// connect_window_signals(window);
	// connect_glarea_signals(glarea);

	gtk_widget_show_all(window);
	do
	{
		gtk_main_iteration();
	} while (gtk_events_pending());
	std::thread t(launchPipeline);
	gtk_main();
	t.join();
}

BOOST_AUTO_TEST_SUITE_END()
