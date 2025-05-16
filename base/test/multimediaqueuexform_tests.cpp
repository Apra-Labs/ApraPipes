
#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include"FileReaderModule.h"
#include "MultimediaQueueXform.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include <ExternalSinkModule.h>
#include "Module.h"
#include "FrameContainerQueue.h"
#include "Mp4WriterSink.h"
#include "EncodedImageMetadata.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "H264Metadata.h"

BOOST_AUTO_TEST_SUITE(multimediaqueuexform_tests)

class SinkModuleProps : public ModuleProps
{
public:
    SinkModuleProps() : ModuleProps()
    {};
};

class SinkModule : public Module
{
public:
    SinkModule(SinkModuleProps props) : Module(SINK, "sinkModule", props)
    {};
    boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
    frame_container pop()
    {
        return Module::pop();
    }

protected:
    bool process() {
      return true;
    };
    bool validateOutputPins()
    {
        return true;
    }
    bool validateInputPins()
    {
        return true;
    }
};

//The multimedia queue takes two arguements queue length (time or no of frames) and bool isDelayInTime. If the bool is true then the length is taken in time else in frames.

int testQueue(uint32_t queuelength, uint16_t tolerance, bool isMapInTime, int i1, int i2, uint64_t startTime, uint64_t endTime)
{
    std::string inFolderPath = "./data/Raw_YUV420_640x360";
    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 20;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
    auto metadata = framemetadata_sp(new RawImageMetadata(640, 360, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto pinId = fileReader->addOutputPin(metadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(queuelength, tolerance, isMapInTime))); // 
    fileReader->setNext(multiQueue);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps())); //

    multiQueue->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(multiQueue->init());
    BOOST_TEST(sink->init());
    auto sinkQueue = sink->getQue();
    for (int i = 0; i < i1; i++)
    {
        fileReader->step();
        multiQueue->step();
    }

    multiQueue->allowFrames(startTime, endTime);
    multiQueue->step();

    for (int i = 0; i < i2; i++)
    {
        fileReader->step();
        multiQueue->step();
    }
    return sinkQueue->size();
}

BOOST_AUTO_TEST_CASE(export_state, *boost::unit_test::disabled())
{
    //In this case both the timestamps (query startTime and query endTime) are in the queue and we pass all the frames requested.

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 1000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 2000;
    endTime = (endTime / 1000) * 1000;

    int queueSize = testQueue(20000, 5000, true, 40, 40, startTime, endTime);
    bool result = (queueSize == 20 || queueSize == 19);
    BOOST_TEST(result);
}

BOOST_AUTO_TEST_CASE(idle_state, *boost::unit_test::disabled())
{
    //In this case both the timestamps (query startTime and endTime) are in the past of the oldest timestamp of queue so state is Idle all the time

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now - 5000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now;
    endTime = (endTime / 1000) * 1000;

    int queueSize = testQueue(10000, 5000, true, 40, 10, startTime, endTime);
    BOOST_TEST(queueSize == 0, "No frames are passed and zero frames are there in queue of Sink");
}

BOOST_AUTO_TEST_CASE(wait_state, *boost::unit_test::disabled())
{
    //In this case both the timestamps (query startTime and endTime) are in the future of the latest timestamp of queue so state is Waiting all the time

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 5000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 8000;
    endTime = (endTime / 1000) * 1000;

    int queueSize = testQueue(10000, 5000, true, 40, 10, startTime, endTime);
    BOOST_TEST(queueSize == 0, "No frames are passed and zero frames are there in queue of Sink");
}

BOOST_AUTO_TEST_CASE(wait_to_export_state, *boost::unit_test::disabled())
{
    //In this case initially we are in wait state then go to export after sometime.

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 2000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 3000;
    endTime = (endTime / 1000) * 1000;

    int queueSize = testQueue(10000, 5000, true, 10, 60, startTime, endTime);
    bool result = (queueSize == 20 || queueSize == 19);
    BOOST_TEST(result);
}

BOOST_AUTO_TEST_CASE(future_export, *boost::unit_test::disabled())
{
    //In this case the timestamp of startTime is in the queue while endTime is in future so we start with export and continue to stay in export as frames are passed.
    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 3500;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 4500;
    endTime = (endTime / 1000) * 1000;

    int queueSize = testQueue(10000, 5000, true, 50, 50, startTime, endTime);
    bool result = (queueSize == 20 || queueSize == 19);
    BOOST_TEST(result);
}

BOOST_AUTO_TEST_CASE(nextQueue_full, *boost::unit_test::disabled())
{
    //In this case, while the frames are being sent to next module the queue of next module must becme full 
    std::string inFolderPath = "./data/Raw_YUV420_640x360";
    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 20;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
    auto metadata = framemetadata_sp(new RawImageMetadata(640, 360, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto pinId = fileReader->addOutputPin(metadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(10000, true)));
    fileReader->setNext(multiQueue);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));

    multiQueue->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(multiQueue->init());
    BOOST_TEST(sink->init());
    auto sinkQueue = sink->getQue();
    for (int i = 0; i < 50; i++)
    {
        fileReader->step();
        multiQueue->step();
    }
    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now - 2000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now;
    endTime = (endTime / 1000) * 1000;
    multiQueue->allowFrames(startTime, endTime);
    multiQueue->step();

    for (int i = 0; i < 20; i++)
    {
        fileReader->step();
        multiQueue->step();
    }

    int queueSize = sinkQueue->size();
    bool result = (queueSize == 20 || queueSize == 19);
    BOOST_TEST(result);
}

BOOST_AUTO_TEST_CASE(prop_change, *boost::unit_test::disabled())
{
    // This testcase is getProps, setProps test - dynamic prop change 
    std::string inFolderPath = "./data/Raw_YUV420_640x360";
    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 20;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
    auto metadata = framemetadata_sp(new RawImageMetadata(640, 360, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
    auto pinId = fileReader->addOutputPin(metadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(10000, 5000, true))); // 
    fileReader->setNext(multiQueue);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps())); //

    multiQueue->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(multiQueue->init());
    BOOST_TEST(sink->init());
    auto sinkQueue = sink->getQue();
    for (int i = 0; i < 40; i++)
    {
        fileReader->step();
        multiQueue->step();
    }

    //We get props and manually change queuelength then set props.
    auto currentProps = multiQueue->getProps();
    currentProps.lowerWaterMark = 12000;
    currentProps.isMapDelayInTime = true;
    auto newValue = MultimediaQueueXformProps(currentProps.lowerWaterMark, 2000, currentProps.isMapDelayInTime);
    multiQueue->setProps(newValue);

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now - 2000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 1000;
    endTime = (endTime / 1000) * 1000;
    multiQueue->allowFrames(startTime, endTime);
    multiQueue->step();

    for (int i = 0; i < 20; i++)
    {
        fileReader->step();
        multiQueue->step();
    }

    int queueSize = sinkQueue->size();
    bool result = (queueSize == 20 || queueSize == 19);
    BOOST_TEST(result);
}

BOOST_AUTO_TEST_CASE(mp4_test_jpeg, *boost::unit_test::disabled())
{
    //In this case we are sending frames from Multimedia Queue to MP4 writer and writing a video 
    //The test is written in run_all_threaded method

    int width = 2048;
    int height = 1536;
    std::string inFolderPath = "./data/re3_filtered";
    std::string outFolderPath = "./data/testOutput/mp4_videos/24bpp/";

    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 24;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
    auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
    auto pinId = fileReader->addOutputPin(encodedImageMetadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(12000, 5000, true))); // 
    fileReader->setNext(multiQueue);
    fileReader->play(true);
    auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 24, outFolderPath);
    mp4WriterSinkProps.logHealth = true;
    mp4WriterSinkProps.logHealthFrequency = 10;
    auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));

    multiQueue->setNext(mp4WriterSink);

    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(fileReader);

    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }

    LOG_INFO << "processing folder <" << inFolderPath << ">";
    p->run_all_threaded();

    Test_Utils::sleep_for_seconds(11);

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now - 10000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 2000;
    endTime = (endTime / 1000) * 1000;
    multiQueue->allowFrames(startTime, endTime);

    Test_Utils::sleep_for_seconds(10);

    p->stop();
    p->term();
    p->wait_for_all();
    p.reset();
}

//H264 Testcases begin here

void testMP4Queue(uint32_t queuelength, uint16_t tolerance, bool isMapInTime, uint64_t startTime, uint64_t endTime)
{
    //In this case we are sending frames from Multimedia Queue to MP4 writer and writing a video
    //The test is written in run_all_threaded method
    int width = 704;
    int height = 576;
    std::string inFolderPath = "./data/h264_data/";
    std::string outFolderPath = "./data/testOutput/mp4_videos/24bpp/";

    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 20;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //

    auto h264ImageMetadata = framemetadata_sp(new H264Metadata(width, height));
    auto pinId = fileReader->addOutputPin(h264ImageMetadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(queuelength, tolerance, true))); //
    fileReader->setNext(multiQueue);

    auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 24, outFolderPath);
    mp4WriterSinkProps.logHealth = true;
    mp4WriterSinkProps.logHealthFrequency = 10;
    auto mp4WriterSink = boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));
    multiQueue->setNext(mp4WriterSink);
    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(fileReader);
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    LOG_ERROR << "processing folder <" << inFolderPath << ">";
    p->run_all_threaded();

    Test_Utils::sleep_for_seconds(20);

    multiQueue->allowFrames(startTime, endTime);

    Test_Utils::sleep_for_seconds(30);

    p->stop();
    p->term();
    p->wait_for_all();
    p.reset();
    std::string videoFolder = { "./data/testOutput/mp4_videos/24bpp" };
    Test_Utils::deleteFolder(videoFolder);
}


BOOST_AUTO_TEST_CASE(mp4_h264_past_export, *boost::unit_test::disabled())
{
    //In this case queryStart is in past and queryEnd is in queue

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 8000;
    endTime = (endTime / 1000) * 1000;
    testMP4Queue(60000, 5000, true, startTime, endTime);

}


BOOST_AUTO_TEST_CASE(mp4_h264_present_export, *boost::unit_test::disabled())
{
    //In this case queryStart and queryEnd both are in queue

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 3000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 10000;
    endTime = (endTime / 1000) * 1000;
    testMP4Queue(60000, 5000, true, startTime, endTime);

}

BOOST_AUTO_TEST_CASE(mp4_h264_present_future_export, *boost::unit_test::disabled())
{
    //In this case queryStart is in queue and queryEnd is in future

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 5000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 30000;
    endTime = (endTime / 1000) * 1000;
    testMP4Queue(10000, 5000, true, startTime, endTime);

}

BOOST_AUTO_TEST_CASE(mp4_h264_future_export, *boost::unit_test::disabled())
{
    //In this case queryStart is in queue and queryEnd is in future

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 30000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 40000;
    endTime = (endTime / 1000) * 1000;
    testMP4Queue(30000, 5000, true, startTime, endTime);

}

BOOST_AUTO_TEST_CASE(fileWriter_test_h264, *boost::unit_test::disabled())
{
    //In this case we are sending frames from Multimedia Queue to file writer and writing the frames
    //The test is written in run_all_threaded method
    int width = 704;
    int height = 576;
    std::string inFolderPath = "./data/h264_data";
    std::string outFolderPath = "./data/testOutput/h264images/";

    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
    fileReaderProps.fps = 20;
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //

    auto h264ImageMetadata = framemetadata_sp(new H264Metadata(width, height));
    auto pinId = fileReader->addOutputPin(h264ImageMetadata);

    auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(60000, 5000, true))); //

    fileReader->setNext(multiQueue);

    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/h264images/Raw_YUV420_640x360????.h264")));
    multiQueue->setNext(fileWriter);

    boost::shared_ptr<PipeLine> p;
    p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
    p->appendModule(fileReader);
    if (!p->init())
    {
        throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    }
    LOG_ERROR << "processing folder <" << inFolderPath << ">";

    p->run_all_threaded();

    Test_Utils::sleep_for_seconds(20);

    boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
    auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
    uint64_t startTime = now + 12000;
    startTime = (startTime / 1000) * 1000;
    uint64_t endTime = now + 17000;
    endTime = (endTime / 1000) * 1000;
    multiQueue->allowFrames(startTime, endTime);

    Test_Utils::sleep_for_seconds(30);

    p->stop();
    p->term();
    p->wait_for_all();
    p.reset();
}


BOOST_AUTO_TEST_SUITE_END()