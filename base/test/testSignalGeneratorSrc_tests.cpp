#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include "RawImageMetadata.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "PipeLine.h"
#include "test_utils.h"
#include "ExternalSinkModule.h"
#include "FrameContainerQueue.h"

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
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(400, 400, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    source->setNext(sink);
    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());
    source->step();
    auto frames = sink->try_pop();
    BOOST_TEST(frames.size() == 1);
    auto outputFrame = frames.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    Test_Utils::saveOrCompare("./data/testOutput/TestSample.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(PropsChange)
{
    auto source = boost::shared_ptr<TestSignalGenerator>(new TestSignalGenerator(TestSignalGeneratorProps(640, 360)));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    source->addOutputPin(metadata);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    source->setNext(sink);
    source->init();
    sink->init();
    source->step();
    auto sinkQue = sink->getQue();
    auto framedata = sinkQue->pop();
    auto frameMetadata = framedata.begin()->second->getMetadata();
    auto size = frameMetadata->getDataSize() / 1.5;
    auto currentProps = source->getProps();
    BOOST_ASSERT(size == currentProps.width * currentProps.height);
    BOOST_TEST(framedata.size() == 1);
    auto outputFrame = framedata.cbegin()->second;
    BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    Test_Utils::saveOrCompare("./data/testOutput/TestSample1.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame->data())), outputFrame->size(), 0);
    TestSignalGeneratorProps propsChange(400, 400);
    source->setProps(propsChange);
    source->step();
    source->step();
    auto secondque = sink->getQue();
    auto framedata2 = secondque->pop();
    auto frameMetadata2 = framedata2.begin()->second->getMetadata();
    auto size2 = frameMetadata2->getDataSize() / 1.5;
    auto currentProps2 = source->getProps();
    BOOST_ASSERT(size2 == currentProps2.width * currentProps2.height);
    BOOST_TEST(framedata2.size() == 1);
    auto outputFrame2 = framedata2.cbegin()->second;
    BOOST_TEST(outputFrame2->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
    Test_Utils::saveOrCompare("./data/testOutput/TestSample2.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(outputFrame2->data())), outputFrame2->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()