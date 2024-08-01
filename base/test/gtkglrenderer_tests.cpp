#include "PipeLine.h"
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>

#if defined(__arm__) || defined(__aarch64__)
#include "DMAFDToHostCopy.h"
#include "EglRenderer.h"
#include "NvArgusCamera.h"
#include "NvTransform.h"
#include "NvV4L2Camera.h"
#include "MemTypeConversion.h"
#include "ResizeNPPI.h"
#include "H264Decoder.h"
#endif
#include "AffineTransform.h"
#include "ColorConversionXForm.h"
#if defined(__arm__)
#include "CudaMemCopy.h"
#endif
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "GtkGlRenderer.h"
#include "H264Metadata.h"
#include "RTSPClientSrc.h"
#include "StatSink.h"
#include "VirtualCameraSink.h"
#include "SimpleControlModule.h"

#include <gtk/gtk.h>

PipeLine p("test");
PipeLine p2("test2");
PipeLine p3("test3");
PipeLine p4("test4");
PipeLine p5("test5");
PipeLine p6("test6");

GtkWidget *glarea;
GtkWidget *glarea2;
GtkWidget *glarea3;
GtkWidget *glarea4;
GtkWidget *glarea5;
GtkWidget *glarea6;
GtkWidget *window;

GtkWidget *glAreaSwitch;
GtkWidget *parentCont;
GtkWidget *parentCont4;
GtkWidget *parentCont3;
GtkWidget *parentCont5;
GtkWidget *parentCont6;

static int pipelineNumber = 0;
string cameraURL = "rtsp://root:pwd@10.102.10.77/axis-media/media.amp";

BOOST_AUTO_TEST_SUITE(gtkglrenderer_tests)

struct rtsp_client_tests_data {
  string outFile;
  string empty;
};

boost::shared_ptr<GtkGlRenderer> GtkGl;

void secondPipeline() {
  p.init();
  p.run_all_threaded();
}

// Below Test is added to Give an Idea about How  Error Callbacks can be used
boost::shared_ptr<GtkGlRenderer> launchErrorCallPipeline() {
  auto fileReaderProps = FileReaderModuleProps("./data/mono_200x200.raw", 0, -1);
  fileReaderProps.readLoop = true;
  fileReaderProps.fps = 300;
  auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
  auto metadata = framemetadata_sp(new RawImageMetadata(200, 200, ImageMetadata::ImageType::MONO, CV_8UC1, 0,
                           CV_8U, FrameMetadata::HOST, true));
  auto rawImagePin = fileReader->addOutputPin(metadata);

  GtkGlRendererProps gtkglsinkProps(glarea, 1, 1);
  GtkGl = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
  fileReader->setNext(GtkGl);

  auto controlProps = SimpleControlModuleProps();
	boost::shared_ptr<SimpleControlModule> mControl = boost::shared_ptr<SimpleControlModule>(new SimpleControlModule(controlProps));


  p.appendModule(fileReader);
  p.addControlModule(mControl);
  p.init();
  mControl->init();
  mControl->enrollModule("Renderer", GtkGl);
  p.run_all_threaded();
  return GtkGl;
}

boost::shared_ptr<GtkGlRenderer> laucX86Pipeline() {
  auto fileReaderProps =
      FileReaderModuleProps("./data/frame_1280x720_rgb.raw", 0, -1);
  fileReaderProps.readLoop = true;
  fileReaderProps.fps = 300;
  auto fileReader = boost::shared_ptr<FileReaderModule>(
      new FileReaderModule(fileReaderProps));
  auto metadata = framemetadata_sp(
      new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0,
                           CV_8U, FrameMetadata::HOST, true));
  auto rawImagePin = fileReader->addOutputPin(metadata);

  GtkGlRendererProps gtkglsinkProps(glarea, 1, 1);
  GtkGl = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
  fileReader->setNext(GtkGl);

  p.appendModule(fileReader);
  p.init();
  p.run_all_threaded();
  return GtkGl;
}

boost::shared_ptr<GtkGlRenderer> laucX86RTSPPipeline() {
#if defined(__arm__) || defined(__aarch64__)
  Logger::setLogLevel("info");

  rtsp_client_tests_data d;
  d.outFile = "./data/testOutput/xyz_???.raw";

  auto url = cameraURL;
  RTSPClientSrcProps rtspProps(url, d.empty, d.empty);
  rtspProps.logHealth = true;
  rtspProps.logHealthFrequency = 100;
  auto rtspSrc = boost::shared_ptr<Module>(new RTSPClientSrc(rtspProps));
  auto meta = framemetadata_sp(new H264Metadata());
  rtspSrc->addOutputPin(meta);

  auto Decoder =
      boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
  rtspSrc->setNext(Decoder);

  auto colorchange = boost::shared_ptr<ColorConversion>(
      new ColorConversion(ColorConversionProps(
          ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB)));
  Decoder->setNext(colorchange);

  GtkGlRendererProps gtkglsinkProps(glarea, 1, 1);
  GtkGl = boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
  colorchange->setNext(GtkGl);

  p.appendModule(rtspSrc);
  p.init();
  p.run_all_threaded();
  return GtkGl;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline1() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d;
  string url = cameraURL;

  // RTSP
  RTSPClientSrcProps rtspProps = RTSPClientSrcProps(url, d.empty, d.empty);
  auto source = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps));
  auto meta = framemetadata_sp(new H264Metadata());
  source->addOutputPin(meta);

  // H264DECODER
  H264DecoderProps decoder_1_Props = H264DecoderProps();
  auto decoder_1 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_1_Props));
  source->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source->setNext(decoder_1);

  // NV-TRANSFORM
  auto transform = boost::shared_ptr<Module>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_1->setNext(transform);

  // //MEMCONVERT TO DEVICE
  auto stream = cudastream_sp(new ApraCudaStream);
  auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(
      MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
  transform->setNext(memconversion1);

  // RESIZE-NPPI
  auto resizenppi = boost::shared_ptr<Module>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream)));
  memconversion1->setNext(resizenppi);

  // MEMCONVERT TO DMA
  auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(
      MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
  resizenppi->setNext(memconversion2);

  GtkGlRendererProps gtkglsinkProps(glarea, 1, 1);
  auto GtkGl =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps));
  memconversion2->setNext(GtkGl);

  p.appendModule(source);
  p.init();
  p.run_all_threaded();
  return GtkGl;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline2() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d2;
  string url2 = "rtsp://10.102.10.75/axis-media/media.amp";

  // RTSP
  RTSPClientSrcProps rtspProps2 = RTSPClientSrcProps(url2, d2.empty, d2.empty);
  auto source2 =
      boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps2));
  auto meta2 = framemetadata_sp(new H264Metadata());
  source2->addOutputPin(meta2);

  // H264DECODER
  H264DecoderProps decoder_1_Props2 = H264DecoderProps();
  auto decoder_12 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_1_Props2));
  source2->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source2->setNext(decoder_12);

  // NV-TRANSFORM
  auto transform2 = boost::shared_ptr<NvTransform>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_12->setNext(transform2);

  // MEMCONVERT TO DEVICE
  auto stream = cudastream_sp(new ApraCudaStream);
  auto memconversion12 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
  transform2->setNext(memconversion12);

  // RESIZE-NPPI
  auto resizenppi2 = boost::shared_ptr<ResizeNPPI>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream)));
  memconversion12->setNext(resizenppi2);

  // MEMCONVERT TO DMA
  auto memconversion22 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
  resizenppi2->setNext(memconversion22);

  GtkGlRendererProps gtkglsinkProps2(glAreaSwitch, 2, 2);
  auto GtkGl2 =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps2));

  memconversion22->setNext(GtkGl2);

  p2.appendModule(source2);
  p2.init();
  p2.run_all_threaded();
  return GtkGl2;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline3() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d3;
  string url3 = "rtsp://10.102.10.42/axis-media/media.amp";

  // RTSP
  RTSPClientSrcProps rtspProps3 = RTSPClientSrcProps(url3, d3.empty, d3.empty);
  auto source3 =
      boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps3));
  auto meta3 = framemetadata_sp(new H264Metadata());
  source3->addOutputPin(meta3);

  // H264DECODER
  H264DecoderProps decoder_3_Props2 = H264DecoderProps();
  auto decoder_13 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_3_Props2));
  source3->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source3->setNext(decoder_13);

  // NV-TRANSFORM
  auto transform3 = boost::shared_ptr<NvTransform>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_13->setNext(transform3);

  // MEMCONVERT TO DEVICE
  auto stream3 = cudastream_sp(new ApraCudaStream);
  auto memconversion13 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream3)));
  transform3->setNext(memconversion13);

  // RESIZE-NPPI
  auto resizenppi3 = boost::shared_ptr<ResizeNPPI>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream3)));
  memconversion13->setNext(resizenppi3);

  // MEMCONVERT TO DMA
  auto memconversion33 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::DMABUF, stream3)));
  resizenppi3->setNext(memconversion33);

  GtkGlRendererProps gtkglsinkProps3(glarea3, 2, 2);
  auto GtkGl3 =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps3));

  memconversion33->setNext(GtkGl3);

  p3.appendModule(source3);
  p3.init();
  p3.run_all_threaded();
  return GtkGl3;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline4() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d4;
  string url4 = "rtsp://10.102.10.42/axis-media/media.amp";

  // RTSP
  RTSPClientSrcProps rtspProps4 = RTSPClientSrcProps(url4, d4.empty, d4.empty);
  auto source4 =
      boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps4));
  auto meta4 = framemetadata_sp(new H264Metadata());
  source4->addOutputPin(meta4);

  // H264DECODER
  H264DecoderProps decoder_4_Props2 = H264DecoderProps();
  auto decoder_14 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_4_Props2));
  source4->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source4->setNext(decoder_14);

  // NV-TRANSFORM
  auto transform4 = boost::shared_ptr<NvTransform>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_14->setNext(transform4);

  // MEMCONVERT TO DEVICE
  auto stream4 = cudastream_sp(new ApraCudaStream);
  auto memconversion14 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream4)));
  transform4->setNext(memconversion14);

  // RESIZE-NPPI
  auto resizenppi4 = boost::shared_ptr<ResizeNPPI>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream4)));
  memconversion14->setNext(resizenppi4);

  // MEMCONVERT TO DMA
  auto memconversion44 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::DMABUF, stream4)));
  resizenppi4->setNext(memconversion44);

  GtkGlRendererProps gtkglsinkProps4(glarea4, 2, 2);
  auto GtkGl4 =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps4));

  memconversion44->setNext(GtkGl4);

  p4.appendModule(source4);
  p4.init();
  p4.run_all_threaded();
  return GtkGl4;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline5() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d5;
  string url5 = "rtsp://10.102.10.75/axis-media/media.amp";

  // RTSP
  RTSPClientSrcProps rtspProps5 = RTSPClientSrcProps(url5, d5.empty, d5.empty);
  auto source5 =
      boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps5));
  auto meta5 = framemetadata_sp(new H264Metadata());
  source5->addOutputPin(meta5);

  // H264DECODER
  H264DecoderProps decoder_5_Props2 = H264DecoderProps();
  auto decoder_15 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_5_Props2));
  source5->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source5->setNext(decoder_15);

  // NV-TRANSFORM
  auto transform5 = boost::shared_ptr<NvTransform>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_15->setNext(transform5);

  // MEMCONVERT TO DEVICE
  auto stream5 = cudastream_sp(new ApraCudaStream);
  auto memconversion15 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream5)));
  transform5->setNext(memconversion15);

  // RESIZE-NPPI
  auto resizenppi5 = boost::shared_ptr<ResizeNPPI>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream5)));
  memconversion15->setNext(resizenppi5);

  // MEMCONVERT TO DMA
  auto memconversion55 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::DMABUF, stream5)));
  resizenppi5->setNext(memconversion55);

  GtkGlRendererProps gtkglsinkProps5(glarea5, 2, 2);
  auto GtkGl5 =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps5));

  memconversion55->setNext(GtkGl5);

  p5.appendModule(source5);
  p5.init();
  p5.run_all_threaded();
  return GtkGl5;
#endif
  return NULL;
}

boost::shared_ptr<GtkGlRenderer> launchPipeline6() {
#if defined(__arm__) || defined(__aarch64__)
  rtsp_client_tests_data d6;
  string url6 = cameraURL;

  // RTSP
  RTSPClientSrcProps rtspProps6 = RTSPClientSrcProps(url6, d6.empty, d6.empty);
  auto source6 =
      boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps6));
  auto meta6 = framemetadata_sp(new H264Metadata());
  source6->addOutputPin(meta6);

  // H264DECODER
  H264DecoderProps decoder_6_Props2 = H264DecoderProps();
  auto decoder_16 =
      boost::shared_ptr<H264Decoder>(new H264Decoder(decoder_6_Props2));
  source6->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
  source6->setNext(decoder_16);

  // NV-TRANSFORM
  auto transform6 = boost::shared_ptr<NvTransform>(
      new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
  decoder_16->setNext(transform6);

  // MEMCONVERT TO DEVICE
  auto stream6 = cudastream_sp(new ApraCudaStream);
  auto memconversion16 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream6)));
  transform6->setNext(memconversion16);

  // RESIZE-NPPI
  auto resizenppi6 = boost::shared_ptr<ResizeNPPI>(
      new ResizeNPPI(ResizeNPPIProps(640, 360, stream6)));
  memconversion16->setNext(resizenppi6);

  // MEMCONVERT TO DMA
  auto memconversion66 =
      boost::shared_ptr<MemTypeConversion>(new MemTypeConversion(
          MemTypeConversionProps(FrameMetadata::DMABUF, stream6)));
  resizenppi6->setNext(memconversion66);

  GtkGlRendererProps gtkglsinkProps6(glarea6, 2, 2);
  auto GtkGl6 =
      boost::shared_ptr<GtkGlRenderer>(new GtkGlRenderer(gtkglsinkProps6));

  memconversion66->setNext(GtkGl6);

  p6.appendModule(source6);
  p6.init();
  p6.run_all_threaded();
  return GtkGl6;
#endif
  return NULL;
}

void screenChanged(GtkWidget *widget, GdkScreen *old_screen,
                   gpointer userdata) {
  /* To check if the display supports alpha channels, get the visual */
  GdkScreen *screen = gtk_widget_get_screen(widget);
  GdkVisual *visual = gdk_screen_get_rgba_visual(screen);
  if (!visual) {
    printf("Your screen does not support alpha channels!\n");
    visual = gdk_screen_get_system_visual(screen);
  } else {
    printf("Your screen supports alpha channels!\n");
  }
  gtk_widget_set_visual(widget, visual);
}

void my_getsize(GtkWidget *widget, GtkAllocation *allocation, void *data) {
  printf("width = %d, height = %d\n", allocation->width, allocation->height);
}

static gboolean hide_gl_area(gpointer data) {
  // gtk_widget_hide(glarea);

  GtkWidget *parentContainer = gtk_widget_get_parent(GTK_WIDGET(glarea));
  gtk_widget_unrealize(glarea);
  // gtk_container_remove(GTK_CONTAINER(parentContainer), glarea);
  // //     // Remove the GtkGLArea from its parent container
  // gtk_gl_area_queue_render(GTK_GL_AREA(glAreaSwitch));

  return G_SOURCE_REMOVE; // Remove the timeout source after execution
}

static gboolean hideWidget(gpointer data) {
  gtk_widget_hide(glarea);
  return G_SOURCE_REMOVE; // Remove the timeout source after execution
}

static gboolean change_gl_area(gpointer data) {
  GtkGl->changeProps(glAreaSwitch, 1280, 720);
  GtkGl->step();
  gtk_container_add(GTK_CONTAINER(parentCont), glAreaSwitch);
  gtk_gl_area_queue_render(GTK_GL_AREA(glAreaSwitch));
  gtk_widget_queue_draw(GTK_WIDGET(glAreaSwitch));

  return G_SOURCE_REMOVE; // Change the glarea before showing
}

static gboolean show_gl_area(gpointer data) {
  // gtk_widget_show(glarea);
  gtk_widget_show(glAreaSwitch);
  return G_SOURCE_REMOVE; // Remove the timeout source after execution
}

void startPipeline6() {
  LOG_ERROR << "CALLING PIPELINE 6!!!!!!";
  launchPipeline6();
  gtk_container_add(GTK_CONTAINER(parentCont6), GTK_WIDGET(glarea6));
  gtk_widget_show(GTK_WIDGET(glarea6));
}

void startPipeline5() {
  LOG_ERROR << "CALLING PIPELINE 5!!!!!!";
  launchPipeline5();
  gtk_container_add(GTK_CONTAINER(parentCont5), GTK_WIDGET(glarea5));
  gtk_widget_show(GTK_WIDGET(glarea5));
}

void startPipeline4() {
  LOG_ERROR << "CALLING PIPELINE 4!!!!!!";
  launchPipeline4();
  gtk_container_add(GTK_CONTAINER(parentCont4), GTK_WIDGET(glarea4));
  gtk_widget_show(GTK_WIDGET(glarea4));
}

void startPipeline3() {
  LOG_ERROR << "CALLING PIPELINE 3!!!!!!";
  launchPipeline3();
  gtk_container_add(GTK_CONTAINER(parentCont3), GTK_WIDGET(glarea3));
  gtk_widget_show(GTK_WIDGET(glarea3));
  // startPipeline4();
}

void on_button_clicked() {
  LOG_ERROR << "CALLING BUTTON CLICKED!!!!!!";
  // gtk_widget_hide(GTK_WIDGET(glarea));
  if (pipelineNumber == 0) {
    launchPipeline2();
    gtk_container_add(GTK_CONTAINER(parentCont), GTK_WIDGET(glAreaSwitch));
    gtk_widget_show(GTK_WIDGET(glAreaSwitch));
  } else if (pipelineNumber == 1) {
    startPipeline3();
  } else if (pipelineNumber == 2) {
    startPipeline4();
  } else if (pipelineNumber == 3) {
    startPipeline5();
  } else if (pipelineNumber == 4) {
    startPipeline6();
  }
  pipelineNumber += 1;
}

BOOST_AUTO_TEST_CASE(windowInit2, *boost::unit_test::disabled()) {
  if (!gtk_init_check(NULL, NULL)) {
    fputs("Could not initialize GTK", stderr);
  }
  GtkBuilder *m_builder = gtk_builder_new();
  if (!m_builder) {
    LOG_ERROR << "Builder not found";
  }
  gtk_builder_add_from_file(m_builder, "./data/app_ui.glade", NULL);

  window = GTK_WIDGET(gtk_window_new(GTK_WINDOW_TOPLEVEL));
  gtk_window_set_decorated(GTK_WINDOW(window), FALSE);
  g_object_ref(window);
  gtk_window_set_default_size(GTK_WINDOW(window), 3840, 2160);
  gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
  gtk_widget_set_app_paintable(window, TRUE);

  do {
    gtk_main_iteration();
  } while (gtk_events_pending());

  GtkWidget *mainFixed =
      GTK_WIDGET(gtk_builder_get_object(m_builder, "A_liveScreen"));
  gtk_container_add(GTK_CONTAINER(window), mainFixed);

  glarea = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw"));
  glAreaSwitch = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw1"));

  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  laucX86Pipeline();
  gtk_widget_show_all(window);

  gtk_main();

  p.stop();
  p.term();
  p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(getErrorCallback, *boost::unit_test::disabled()) {
  if (!gtk_init_check(NULL, NULL)) {
    fputs("Could not initialize GTK", stderr);
  }
  GtkBuilder *m_builder = gtk_builder_new();
  if (!m_builder) {
    LOG_ERROR << "Builder not found";
  }
  gtk_builder_add_from_file(m_builder, "./data/app_ui.glade", NULL);

  window = GTK_WIDGET(gtk_window_new(GTK_WINDOW_TOPLEVEL));
  gtk_window_set_decorated(GTK_WINDOW(window), FALSE);
  g_object_ref(window);
  gtk_window_set_default_size(GTK_WINDOW(window), 1920, 1080);
  gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
  gtk_widget_set_app_paintable(window, TRUE);

  do {
    gtk_main_iteration();
  } while (gtk_events_pending());

  GtkWidget *mainFixed =
      GTK_WIDGET(gtk_builder_get_object(m_builder, "A_liveScreen"));
  gtk_container_add(GTK_CONTAINER(window), mainFixed);

  glarea = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw"));
  glAreaSwitch = GTK_WIDGET(gtk_builder_get_object(m_builder, "glareadraw1"));

  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  launchErrorCallPipeline();
  gtk_widget_show_all(window);

  gtk_main();

  p.stop();
  p.term();
  p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
