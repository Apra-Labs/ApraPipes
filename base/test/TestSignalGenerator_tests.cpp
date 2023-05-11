#include "TestSignalGenerator.h"
#include "Module.h"
#include "RawImageMetadata.h"
#include <boost/test/unit_test.hpp>
#include "FileWriterModule.h"
#include <boost/filesystem.hpp>
#include <PipeLine.h>
#include <ImageViewerModule.h>
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(TestSignalGenerator_tests)

BOOST_AUTO_TEST_CASE(basic)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps()));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);
    auto sink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("Test-signal")));
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


BOOST_AUTO_TEST_CASE(file)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps()));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);

    auto sink = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/test.raw")));
    source->setNext(sink);

    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    
    source->step();
    sink->step();
}


BOOST_AUTO_TEST_CASE(sink)
{
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