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
    auto size = frameMetadata->getDataSize() / 1.5;
    auto currentProps = source->getProps();
    BOOST_ASSERT(size == currentProps.width * currentProps.height);
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
    size = frameMetadata->getDataSize() / 1.5;
    BOOST_TEST(frames.size() == 1);
    outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    pReadDataTest = const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data()));
    readDataSizeTest = outputFrame->size();
    Test_Utils::saveOrCompare("./data/TestSample2.raw",pReadDataTest,readDataSizeTest, 0);
}

BOOST_AUTO_TEST_SUITE_END()