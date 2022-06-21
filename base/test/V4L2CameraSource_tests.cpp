#include <boost/test/unit_test.hpp>
 
#include "NvArgusCamera.h"
#include "NvTransform.h"
#include "FileWriterModule.h"
#include "H264EncoderV4L2.h"
#include "VirtualCameraSink.h"
#include "RTSPPusher.h"
#include "StatSink.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "DMAFDToHostCopy.h"
#include "test_utils.h"
#include "V4L2CameraSource.h"
#include "HostDMA.h"
#include "BayerToYUV420A.h"
#include "CudaMemCopy.h"
#include "DeviceToDMA.h"
#include "ImageResizeCV.h"
#include "ResizeNPPI.h"
#include "BayerToGray.h"
#include "RestrictCapFrames.h"
#include "DeviceToDMAMono.h"
#include "BayerToMono.h"
 
BOOST_AUTO_TEST_SUITE(V4L2CameraSource_tests)
 
BOOST_AUTO_TEST_CASE(v4l2camsaveqq, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps1(800, 800, "/dev/video0");
    sourceProps1.maxConcurrentFrames = 10;
    sourceProps1.fps = 60;
    auto source1 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps1));
 
    auto filewriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./V4L2Output/ImageA?????.raw", true, false)));
    source1->setNext(filewriter);
 
    PipeLine p("test");
    p.appendModule(source1);
    BOOST_TEST(p.init());
 
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
 
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(2));
 
    //source1->resetFrameCapture();
    boost::this_thread::sleep_for(boost::chrono::seconds(2000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_CASE(camerawriteAllExternal, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));
 
    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameA????.raw", false, false)));
    source->setNext(fileWriter);
 
    PipeLine p("test");
    p.appendModule(source);
    BOOST_TEST(p.init());
 
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
 
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(10000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);
 
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_CASE(cameraSinkAll, *boost::unit_test::disabled())
{
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    // Logger::setLogLevel(boost::log::trivial::severity_level::error);
 
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));
 
    StatSinkProps sinkProps;
    sinkProps.logHealth = true;
    sinkProps.logHealthFrequency = 100;
    auto sink1 = boost::shared_ptr<Module>(new StatSink(sinkProps));
 
    source->setNext(sink1);
    PipeLine p("test");
    p.appendModule(source);
 
    BOOST_TEST(p.init());
 
   
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(10000));
   
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_CASE(camerawriteAll, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));
 
    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frameA????.raw", false, false)));
    source->setNext(fileWriter);
 
    PipeLine p("test");
    p.appendModule(source);
    BOOST_TEST(p.init());
 
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
 
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(10000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);
 
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_CASE(camerawrite, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));
 
    BayerToYUV420Props yuv420props;
    yuv420props.qlen = 1;
    auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(yuv420props));
    source->setNext(bayerTRGBA);
 
    auto stream = cudastream_sp(new ApraCudaStream);
 
    DeviceToDMAProps deviceTodmaprops(stream);
    deviceTodmaprops.qlen = 1;
    deviceTodmaprops.logHealth = true;
    deviceTodmaprops.logHealthFrequency = 100;
 
    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/data/testOutput/frameA????.raw", false, true)));
    source->setNext(fileWriter);
 
    PipeLine p("test");
    p.appendModule(source);
    BOOST_TEST(p.init());
 
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
 
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(1000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);
 
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_CASE(v4l2camRenderWrite, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));
 
    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("/media/nvidia/Madhav/CameraRenderWrite/testOutput/frameA????.raw", false, false)));
    source->setNext(fileWriter);
 
    BayerToYUV420Props yuv420props;
    yuv420props.qlen = 1;
    auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(yuv420props));
    source->setNext(bayerTRGBA);
 
    auto stream = cudastream_sp(new ApraCudaStream);
 
    DeviceToDMAProps deviceTodmaprops(stream);
    deviceTodmaprops.qlen = 1;
    deviceTodmaprops.logHealth = true;
    deviceTodmaprops.logHealthFrequency = 100;
 
    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
    devicedma->setNext(sink);
 
    PipeLine p("test");
    p.appendModule(source);
    BOOST_TEST(p.init());
 
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
 
    p.run_all_threaded();
 
    boost::this_thread::sleep_for(boost::chrono::seconds(10000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);
 
    p.stop();
    p.term();
 
    p.wait_for_all();
}
 
BOOST_AUTO_TEST_SUITE_END()
