#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "ResizeNPPI.h"
#include "test_utils.h"
#include "NvArgusCamera.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include "ResizeNPPIDMA.h"
#include "NvTransform.h"

BOOST_AUTO_TEST_SUITE(resizenppidma_tests)

BOOST_AUTO_TEST_CASE(render)
{
    auto width = 1280;
    auto height = 720;

    NvArgusCameraProps sourceProps(width, height, 0);
    sourceProps.maxConcurrentFrames = 10;
    sourceProps.fps = 60;
    auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

    auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
    source->setNext(nv_transform);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto m2 = boost::shared_ptr<Module>(new ResizeNPPIDMA(ResizeNPPIDMAProps(width >> 1, height >> 1, stream)));
    nv_transform->setNext(m2);

    EglRendererProps renderProps(0, 0, 400, 400);
    renderProps.logHealth = true;
    renderProps.qlen = 1;
    renderProps.logHealthFrequency = 100;
    renderProps.quePushStrategyType = QuePushStrategy::QuePushStrategyType::NON_BLOCKING_ANY;
    auto renderer = boost::shared_ptr<Module>(new EglRenderer(renderProps));
    m2->setNext(renderer);

    PipeLine p("test");
    p.appendModule(source);
    BOOST_TEST(p.init());
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(1000));
    Logger::setLogLevel(boost::log::trivial::severity_level::error);

    p.stop();
    p.term();
    p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
