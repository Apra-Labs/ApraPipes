#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include "RawImageMetadata.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "PipeLine.h"
#include "test_utils.h"
#include "ExternalSinkModule.h"
#include "FrameContainerQueue.h"
#include"FileWriterModule.h"

BOOST_AUTO_TEST_SUITE(TestSignalGenerator_tests)

class SinkModuleProps : public ModuleProps
{
public:
    SinkModuleProps() : ModuleProps(){};
};

class SinkModule : public Module
{
public:
    SinkModule(SinkModuleProps props) : Module(SINK, "sinkModule", props){};
    boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }

protected:
    bool validateOutputPins()
    {
        return true;
    }
    bool validateInputPins()
    {
        return true;
    }
};

BOOST_AUTO_TEST_CASE(Basic)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps(400, 400)));
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    source->step();
    auto frames = sink->try_pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    const uint8_t* pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    unsigned int readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample.raw", pReadDataTest, readDataSizeTest,0);
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps(640, 360)));
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    source->setNext(sink);
    source->init();
    sink->init();
    source->step();
    auto sinkQue = sink->getQue();
    frame_container frames;
    frames = sinkQue->pop();
    auto frameMetadata = frames.begin()->second->getMetadata();
    auto currentProps = source->getProps();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    const uint8_t* pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    unsigned int readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample1.raw", pReadDataTest,readDataSizeTest, 0);
    TestSignalGeneratorProps newProps(400, 400);
    source->setProps(newProps);
    source->step();
    sinkQue = sink->getQue();
    frames = sinkQue->pop();
    frameMetadata = frames.begin()->second->getMetadata();
    BOOST_TEST(frames.size() == 1);
    outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample2.raw",pReadDataTest,readDataSizeTest, 0);
}

BOOST_AUTO_TEST_CASE(FrameIndexOverlay)
{
    TestSignalGeneratorProps props(640, 360);
    props.overlayType = OverlayType::FRAME_INDEX;
    props.overlayFgColor = "00FF00";  // Green
    props.overlayBgColor = "000000";  // Black
    
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(props));
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    
    // Generate 5 frames
    for (int i = 0; i < 5; i++)
    {
        source->step();
        auto frames = sink->try_pop();
        BOOST_TEST(frames.size() == 1);
        auto outputFrame = frames.cbegin()->second;
        BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
        const uint8_t* pData = static_cast<const uint8_t*>(outputFrame->data());
        unsigned int dataSize = outputFrame->size();
        std::string filename = "./data/frame_overlay_" + std::to_string(i) + ".raw";
        Test_Utils::saveOrCompare(filename.c_str(), pData, dataSize, 0);
    }
}

BOOST_AUTO_TEST_CASE(TimestampOverlay)
{
    TestSignalGeneratorProps props(640, 360);
    props.overlayType = OverlayType::TIMESTAMP;
    props.timestampFormat = "%H:%M:%S";  // hh:mm:ss format
    
   auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(props));
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    
    source->step();
    auto frames = sink->try_pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    
    
    const uint8_t* pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    unsigned int readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample_Timestamp.raw", pReadDataTest, readDataSizeTest, 0);
}

BOOST_AUTO_TEST_CASE(BothOverlays)
{
    TestSignalGeneratorProps props(640, 360);
    props.overlayType = OverlayType::BOTH;
    props.overlayFgColor = "FFFF00";  // Yellow
    props.overlayBgColor = "000080";  // Dark blue
    
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(props));
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    
    source->step();
    auto frames = sink->try_pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    
    const uint8_t* pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    unsigned int readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample_Both.raw", pReadDataTest, readDataSizeTest, 0);
}

BOOST_AUTO_TEST_CASE(CustomOverlayConfig)
{
    TestSignalGeneratorProps props(800, 600);
    props.overlayType = OverlayType::FRAME_INDEX;
    props.overlayFgColor = "FF0000";  
    props.overlayBgColor = "FFFFFF";            
    props.overlayFontSize = 2.0; 
    props.overlayX = 50;
    props.overlayY = 100;  
    
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(props));
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    
    source->step();
    auto frames = sink->try_pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    const uint8_t* pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    unsigned int readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample_CustomOverlay.raw", pReadDataTest, readDataSizeTest, 0);
}

BOOST_AUTO_TEST_SUITE_END()