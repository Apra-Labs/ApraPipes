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

BOOST_AUTO_TEST_SUITE(sound_record_tests)

BOOST_AUTO_TEST_CASE(record)
{
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    SoundRecordProps sourceProps(48000,1,2,0,200);
    auto source = boost::shared_ptr<Module>(new SoundRecord(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("../test.wav", true)));
    source->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    p.stop();
    p.term();
    p.wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()