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
#include "BayerToRGBA.h"
#include "CudaMemCopy.h"
#include "DeviceToDMA.h"
#include "ImageResizeCV.h"
#include "ResizeNPPI.h"
#include "BayerToGray.h"

BOOST_AUTO_TEST_SUITE(camerav4l2_3_tests)

BOOST_AUTO_TEST_CASE(v4l2camSingleRender, *boost::unit_test::disabled())
{
    V4L2CameraSourceProps sourceProps(800, 800, "/dev/video0");
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps));

    BayerToRGBAProps brgbprops;
    brgbprops.qlen = 1;
    auto bayerTRGBA = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops));
    source->setNext(bayerTRGBA);

    auto stream = cudastream_sp(new ApraCudaStream);

    DeviceToDMAProps deviceTodmaprops(stream);
    deviceTodmaprops.qlen = 1;
    deviceTodmaprops.logHealth = true;
    deviceTodmaprops.logHealthFrequency = 100;

    auto devicedma = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops));
    bayerTRGBA->setNext(devicedma);

    auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
    devicedma->setNext(sink);

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

BOOST_AUTO_TEST_CASE(v4l2camAlleRender, *boost::unit_test::disabled())
{
    auto stream = cudastream_sp(new ApraCudaStream);

    V4L2CameraSourceProps sourceProps1(800, 800, "/dev/video0");
    sourceProps1.maxConcurrentFrames = 10;
    sourceProps1.fps = 60;
    auto source1 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps1));

    BayerToRGBAProps brgbprops1;
    brgbprops1.qlen = 1;
    auto bayerTRGBA1 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops1));
    source1->setNext(bayerTRGBA1);

    DeviceToDMAProps deviceTodmaprops1(stream);
    deviceTodmaprops1.qlen = 1;
    deviceTodmaprops1.logHealth = true;
    deviceTodmaprops1.logHealthFrequency = 100;

    auto devicedma1 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops1));
    bayerTRGBA1->setNext(devicedma1);

    auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0, 800, 800)));
    devicedma1->setNext(sink1);

    /// Camera 2


    V4L2CameraSourceProps sourceProps2(800, 800, "/dev/video1");
    sourceProps2.maxConcurrentFrames = 10;
    sourceProps2.fps = 60;
    auto source2 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps2));

    BayerToRGBAProps brgbprops2;
    brgbprops2.qlen = 1;
    auto bayerTRGBA2 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops2));
    source2->setNext(bayerTRGBA2);

    DeviceToDMAProps deviceTodmaprops2(stream);
    deviceTodmaprops2.qlen = 1;
    deviceTodmaprops2.logHealth = true;
    deviceTodmaprops2.logHealthFrequency = 100;

    auto devicedma2 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops2));
    bayerTRGBA2->setNext(devicedma2);

    auto sink2 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(800, 0, 800, 800)));
    devicedma2->setNext(sink2);

    /// camera 3

    V4L2CameraSourceProps sourceProps3(800, 800, "/dev/video2");
    sourceProps3.maxConcurrentFrames = 10;
    sourceProps3.fps = 60;
    auto source3 = boost::shared_ptr<Module>(new V4L2CameraSource(sourceProps3));

    BayerToRGBAProps brgbprops3;
    brgbprops3.qlen = 1;
    auto bayerTRGBA3 = boost::shared_ptr<Module>(new BayerToRGBA(brgbprops3));
    source3->setNext(bayerTRGBA3);

    DeviceToDMAProps deviceTodmaprops3(stream);
    deviceTodmaprops3.qlen = 1;
    deviceTodmaprops3.logHealth = true;
    deviceTodmaprops3.logHealthFrequency = 100;

    auto devicedma3 = boost::shared_ptr<Module>(new DeviceToDMA(deviceTodmaprops3));
    bayerTRGBA3->setNext(devicedma3);

    auto sink3 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 800, 800, 800)));
    devicedma3->setNext(sink3);

    PipeLine p("test");
    p.appendModule(source1);
    p.appendModule(source2);
    p.appendModule(source3);

    BOOST_TEST(p.init());

    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    p.run_all_threaded();

    boost::this_thread::sleep_for(boost::chrono::seconds(1000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);

    p.stop();
    p.term();

    p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
