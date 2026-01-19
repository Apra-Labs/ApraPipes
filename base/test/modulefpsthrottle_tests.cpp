#include <stdafx.h>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Module.h"
#include "FileReaderModule.h"
#include "ValveModule.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "EncodedImageMetadata.h"

BOOST_AUTO_TEST_SUITE(modulefpsthrottle_tests)

BOOST_AUTO_TEST_CASE(throttle_transform_fps)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    auto readerProps = FileReaderModuleProps("./data/mono_1920x960.jpg");
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(readerProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    readerProps.readLoop = true;
    readerProps.logHealth = true;
    readerProps.logHealthFrequency = 50;
    fileReader->addOutputPin(metadata);
    
    auto valveProps = ValveModuleProps(-1);
    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(valveProps));
    valveProps.logHealthFrequency = 100;
    fileReader->setNext(valve);
    auto valveMetadata = framemetadata_sp(new RawImageMetadata());
    auto rawImagePin = valve->addOutputPin(valveMetadata);
    
    StatSinkProps sinkProps;
    sinkProps.logHealthFrequency = 100;
    auto m3 = boost::shared_ptr<StatSink>(new StatSink(sinkProps));
    valve->setNext(m3);
    
    //Pipeline
    PipeLine p("throttle_fps_test");
    p.appendModule(fileReader);
    p.init();
    p.run_all_threaded();
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    boost::this_thread::sleep_for(boost::chrono::seconds(20));
    
    auto sinkFPS = m3->getPipelineFps();
    LOG_INFO << " Transform fps initial value is : "<< sinkFPS;
    //Throttling Transform fps to 25 and verifying fps change in sinkFPS
    LOG_INFO << "Throttling Transform fps to 25.";
    valve->tryThrottlingFPS(25);
    boost::this_thread::sleep_for(boost::chrono::seconds(30));
    sinkFPS = m3->getPipelineFps();
    bool testFPS = sinkFPS > 23 && sinkFPS < 27;
    BOOST_TEST(testFPS);
    LOG_INFO << "Transform FPS is " << sinkFPS << ".";
    
    //Throttling Transform fps to 10
    LOG_INFO << "Throttling Transform fps to 10.";
    valve->tryThrottlingFPS(10);
    boost::this_thread::sleep_for(boost::chrono::seconds(30));
    sinkFPS = m3->getPipelineFps();
    testFPS = sinkFPS > 8 && sinkFPS < 12;
    BOOST_TEST(testFPS);
    LOG_INFO << "Transform FPS is " << sinkFPS << ".";
    
    //Throttling Transform fps to 40
    LOG_INFO << "Throttling Transform fps to 40.";
    valve->tryThrottlingFPS(40);
    boost::this_thread::sleep_for(boost::chrono::seconds(30));
    sinkFPS = m3->getPipelineFps();
    testFPS = sinkFPS > 38 && sinkFPS < 42;
    BOOST_TEST(testFPS);
    LOG_INFO << "Transform FPS is " << sinkFPS << ".";
    p.stop();
    p.term();
    p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(throttle_fps_branch)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    auto readerProps = FileReaderModuleProps("./data/mono_1920x960.jpg");
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(readerProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    readerProps.readLoop = true;
    readerProps.logHealth = true;
    readerProps.logHealthFrequency = 50;
    fileReader->addOutputPin(metadata);

    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(-1)));
    fileReader->setNext(valve);
    StatSinkProps sink1Props;
    sink1Props.logHealthFrequency = 250;
    auto sink1 = boost::shared_ptr<StatSink>(new StatSink(sink1Props));
    fileReader->setNext(sink1);
    StatSinkProps sink2Props;
    sink2Props.logHealthFrequency = 250;
    auto sink2 = boost::shared_ptr<StatSink>(new StatSink(sink2Props));
    valve->setNext(sink2);
    StatSinkProps sink3Props;
    sink3Props.logHealthFrequency = 250;
    auto sink3 = boost::shared_ptr<StatSink>(new StatSink(sink3Props));
    valve->setNext(sink3);
    
    PipeLine p("throttle_fps_test");
    p.appendModule(fileReader);
    p.init();
    p.run_all_threaded();
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    boost::this_thread::sleep_for(boost::chrono::seconds(20));
    
    //Checking if Transform FPS is 50 for both sinks
    auto sink2FPS = sink2->getPipelineFps();
    LOG_INFO << "Initial Transform FPS according to Sink2 is " << sink2FPS << ".";
    auto sink3FPS = sink3->getPipelineFps();
    LOG_INFO << "Initial Transform FPS according to Sink3 is " << sink3FPS << ".";
    
    //Throttling Transform fps to 40 and verifying fps change in sinkFPS
    LOG_INFO << "Throttling Transform fps to 40.";
    valve->tryThrottlingFPS(40);
    boost::this_thread::sleep_for(boost::chrono::seconds(60));
    sink2FPS = sink2->getPipelineFps();
    auto testFPS = sink2FPS > 39 && sink2FPS < 41;
    BOOST_TEST(testFPS);
    LOG_INFO << "Transform FPS according to Sink2 is " << sink2FPS << ".";
    sink3FPS = sink3->getPipelineFps();
    testFPS = sink3FPS > 39 && sink3FPS < 41;
    BOOST_TEST(testFPS);
    LOG_INFO << "Transform FPS according to Sink3 is " << sink3FPS << ".";
}

BOOST_AUTO_TEST_SUITE_END()