#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "SoundRecord.h"
#include "ExternalSinkModule.h"
#include "Module.h"

BOOST_AUTO_TEST_SUITE(sound_record_tests)

BOOST_AUTO_TEST_CASE(recordMono, *boost::unit_test::disabled())
{
    // Manual test, listen to the file on audacity to for sanity check
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto time_to_run = Test_Utils::getArgValue("s", "10");
    auto n_seconds = atoi(time_to_run.c_str());

    SoundRecordProps sourceProps(48000,1,0,200);
    auto source = boost::shared_ptr<Module>(new SoundRecord(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("../testMono.wav", true)));
    source->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(n_seconds));
    p.stop();
    p.term();
    p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(recordStereo, *boost::unit_test::disabled())
{
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto time_to_run = Test_Utils::getArgValue("s", "10");
    auto n_seconds = atoi(time_to_run.c_str());

    SoundRecordProps sourceProps(48000,2,0,200);
    auto source = boost::shared_ptr<SoundRecord>(new SoundRecord(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("../testStereo.wav", true)));
    source->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(n_seconds));
    p.stop();
    p.term();
    p.wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()