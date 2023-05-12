#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include "RawImageMetadata.h"
#include <boost/test/unit_test.hpp>
#include "FileWriterModule.h"
#include <boost/filesystem.hpp>
#include "PipeLine.h"
#include "test_utils.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(TestSignalGenerator_tests)

BOOST_AUTO_TEST_CASE(basic)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps()));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);
    auto sink = boost::shared_ptr<Module>(new StatSink());	
    source->setNext(sink);
    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(source);
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    p->run_all_threaded();
    Test_Utils::sleep_for_seconds(2);
    p->stop();
    p->term();
    p->wait_for_all();
    p.reset();
}


BOOST_AUTO_TEST_CASE(fileWriterSink)
{
    std::vector<std::string> Files = {"./data/testsample1.raw"};
    Test_Utils::FileCleaner f(Files);
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps()));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);

    auto sink = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testsample1.raw")));
    source->setNext(sink);

    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(source);
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    p->run_all_threaded();
    Test_Utils::sleep_for_seconds(1);
    p->stop();
    p->term();
    p->wait_for_all();
    p.reset();
}



BOOST_AUTO_TEST_SUITE_END()     