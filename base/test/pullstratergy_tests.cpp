#include <boost/test/unit_test.hpp>
#include <memory>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "PipeLine.h"
#include "StatSink.h"
#include "Module.h"
#include "H264Metadata.h"

#include <chrono>
#include <thread>
#include <iostream>

BOOST_AUTO_TEST_SUITE(pullstratergy_tests)

class PullAnalogy : public Module
{
public:
    PullAnalogy(ModuleProps _props) : Module(SINK, "PullAnalogy", _props)
    {
    }

    virtual ~PullAnalogy() {}

    bool step()
    {
        return Module::step();
    }

protected:
    bool process(frame_container &frames)
    {
        LOG_INFO << "HOLA SOMEONE PULLED ME";
        lastFrame = frames.cbegin()->second;
        LOG_INFO << "Frame Size is :" << lastFrame->size();
        return true;
    }

    bool validateInputPins()
    {
        return true;
    }

    frame_sp lastFrame;
};

void pull(std::shared_ptr<PullAnalogy> sink)
{
    // call getFrame here in a loop - let's say 10 times
    for (int i = 0; i < 10; i++)
    {
        sink->step();
    }
    return;
};

BOOST_AUTO_TEST_CASE(pullAnalogy, *boost::unit_test::disabled())
{

    LoggerProps logprops;
    logprops.logLevel = boost::log::trivial::severity_level::info;
    Logger::initLogger(logprops);

    // metadata is known
    auto width = 640;
    auto height = 360;

    FileReaderModuleProps fileReaderProps("./data/h264_frames/Raw_YUV420_640x360_????.h264");
    // fileReaderProps.fps = 30;
    auto fileReader = std::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
    //create a class of h264Metadata
    auto metadata = framemetadata_sp(new H264Metadata(width, height));
    fileReader->addOutputPin(metadata);

    ModuleProps _props;
    _props.frameFetchStrategy = ModuleProps::FrameFetchStrategy::PULL;
    _props.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
    _props.qlen = 1;

    auto sink = std::shared_ptr<PullAnalogy>(new PullAnalogy(_props));
    fileReader->setNext(sink);

    PipeLine p("test");
    p.appendModule(fileReader);
    BOOST_TEST(p.init());
    p.run_all_threaded();

    std::thread t1(pull, sink);
    t1.join();
    p.stop();
    p.wait_for_all(true);
    p.term();
   
}

BOOST_AUTO_TEST_SUITE_END()
