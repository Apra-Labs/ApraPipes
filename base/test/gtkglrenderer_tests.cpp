#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "VirtualCameraSink.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "StatSink.h"
#include "ResizeNPPI.h"
#include "AffineTransform.h"
#include "H264Decoder.h"
#include "CudaMemCopy.h"
#include "H264Metadata.h"
#include "RTSPClientSrc.h"
#include "EglRenderer.h"
#include "GtkGlRenderer.h"
#include "FileWriterModule.h"
#include "NvArgusCamera.h"
#include "MemTypeConversion.h"
#include <gtk/gtk.h>
// // #include <constants/GlobalProperties.h>
// #include <utils/GTK_UI.h>
// #define PRIMARY_WINDOW_WIDTH 	1920
// #define PRIMARY_WINDOW_HEIGHT 	1080

// #define ASSETS_PATH "assets_ui/"
// #define GLADE_PATH ASSETS_PATH "ui/"
// #define STYLE_PATH ASSETS_PATH "css/"
// #define CONFIG_PATH "config/"

PipeLine p("test");
GtkWidget *glarea;
GtkWidget *glarea2;
GtkWidget *glarea3;
GtkWidget *glarea4;
BOOST_AUTO_TEST_SUITE(gtkglrenderer_tests)

struct rtsp_client_tests_data {
	string outFile;
	string empty;
};

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

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 360, 10)));	
	
	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(transform);

	// GtkGlRendererProps gtkglsinkProps(glarea, 1280, 720);
	// auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	// transform->setNext(sink);
	
	// GtkGlRendererProps gtkglsinkProps2(glarea2, 1280, 720);
	// auto sink2 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps2));
	// transform->setNext(sink2);

	// GtkGlRendererProps gtkglsinkProps3(glarea3, 1280, 720);
	// auto sink3 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps3));
	// transform->setNext(sink3);

	// GtkGlRendererProps gtkglsinkProps4(glarea4, 1280, 720);
	// auto sink4 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps4));
	// transform->setNext(sink4);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0)));																																																															
	transform->setNext(sink);

	p.appendModule(source);
	p.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
}
boost::shared_ptr<GtkGlRenderer> launchPipeline()
{
	rtsp_client_tests_data d;
	string url = "rtsp://root:m4m1g0@10.102.10.77/axis-media/media.amp";

	//RTSP
	RTSPClientSrcProps rtspProps = RTSPClientSrcProps(url, d.empty, d.empty);
	auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps));
	auto meta = framemetadata_sp(new H264Metadata());
	source->addOutputPin(meta);

	//H264DECODER
	H264DecoderProps decoder_1_Props = H264DecoderProps();
	auto decoder_1 = boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_1_Props));
	source->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	source->setNext(decoder_1);

	//NV-TRANSFORM
	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	decoder_1->setNext(transform);

	//MEMCONVERT TO DEVICE
	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	transform->setNext(memconversion1);

	//RESIZE-NPPI
	auto resizenppi = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(1280, 720, stream)));
	memconversion1->setNext(resizenppi);

	//MEMCONVERT TO DMA
	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	resizenppi->setNext(memconversion2);

	GtkGlRendererProps gtkglsinkProps(glarea, 1280, 720);
	auto sink = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
	memconversion2->setNext(sink);

	auto eglsink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));																																																															
	// memconversion2->setNext(eglsink);
	
	// GtkGlRendererProps gtkglsinkProps2(glarea2, 1024, 1024);
	// auto sink2 = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps2));
	// memconversion2->setNext(sink2);

	// GtkGlRendererProps gtkglsinkProps3(glarea3, 1024, 1024);
	// auto sink3 = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps3));
	// memconversion2->setNext(sink3);

	// GtkGlRendererProps gtkglsinkProps4(glarea4, 1024, 1024);
	// auto sink4 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps4));
	// memconversion2->setNext(sink4);

	// auto eglsink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));																																																															
	// decoder_1->setNext(eglsink);

	p.appendModule(source);
	p.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();

	return sink;																																																															
;

}

void launchPipelineRTSP()
{
	rtsp_client_tests_data d;
	string url = "rtsp://10.102.10.77/axis-media/media.amp";
	RTSPClientSrcProps rtspProps = RTSPClientSrcProps(url, d.empty, d.empty);
	auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps));
	auto meta = framemetadata_sp(new H264Metadata());
	source->addOutputPin(meta);
	
	H264DecoderProps decoder_1_Props = H264DecoderProps();
	auto decoder_1 = boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_1_Props));
	source->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	source->setNext(decoder_1);



	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	decoder_1->setNext(transform);

	// auto stream = cudastream_sp(new ApraCudaStream);
	// auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	// transform->setNext(copy1);

	// auto m2 = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(640, 360, stream)));
	// copy1->setNext(m2);
	// auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	// m2->setNext(copy2);
	// auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];


	GtkGlRendererProps gtkglsinkProps(glarea, 1280, 720);
	auto sink = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps));
	transform->setNext(sink);
	
	GtkGlRendererProps gtkglsinkProps2(glarea2, 1280, 720);
	auto sink2 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps2));
	transform->setNext(sink2);

	GtkGlRendererProps gtkglsinkProps3(glarea3, 1280, 720);
	auto sink3 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps3));
	transform->setNext(sink3);

	GtkGlRendererProps gtkglsinkProps4(glarea4, 1280, 720);
	auto sink4 = boost::shared_ptr<Module>(new GtkGlRenderer(gtkglsinkProps4));
	transform->setNext(sink4);

	// auto eglsink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));																																																															
	// decoder_1->setNext(eglsink);

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

void my_getsize(GtkWidget *widget, GtkAllocation *allocation, void *data) {
    printf("width = %d, height = %d\n", allocation->width, allocation->height);
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
	gtk_builder_add_from_file(m_builder, "/mnt/disks/ssd/vinayak/backup/GtkRendererModule/ApraPipes/assets/appui.glade", NULL);
	std::cout << "ui glade found" << std::endl;

	GtkWidget *window = GTK_WIDGET(gtk_window_new(GTK_WINDOW_TOPLEVEL));
	g_object_ref(window);
	gtk_window_set_default_size(GTK_WINDOW(window), 2048, 2048);
	gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
	gtk_widget_set_app_paintable(window, TRUE);

	do
	{
		gtk_main_iteration();
	} while (gtk_events_pending());

	GtkWidget *mainFixed = GTK_WIDGET(gtk_builder_get_object(m_builder, "mainWidget"));
	gtk_container_add(GTK_CONTAINER(window), mainFixed);
	glarea = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw"));
	// glarea2 = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw2"));
	// glarea3 = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw3"));
	// glarea4 = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw4"));
	g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL); 
	g_signal_connect(glarea, "size-allocate", G_CALLBACK(my_getsize), NULL);
	launchPipeline();
	gtk_widget_show_all(window);
	gtk_main();

	p.stop();
	p.term();
	p.wait_for_all();
}



BOOST_AUTO_TEST_SUITE_END()
