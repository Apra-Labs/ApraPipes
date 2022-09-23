#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "ValveModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "Module.h"

BOOST_AUTO_TEST_SUITE(valveModule_tests)

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

protected:
    bool process() {return false;}
    bool validateOutputPins()
    {
        return true;
    }
    bool validateInputPins()
    {
        return true;
    }
};

BOOST_AUTO_TEST_CASE(basic)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->addOutputPin(metadata);
    valve->setNext(sink);
     

    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));
   
    // We are sending 4 frames with enable false so only no frames pass through the valve
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 0);
    //We are resetting the props to 4 by command and enable true 
    valve->allowFrames(4);
    valve->step();

    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 4);

    //The props are changed by passing 2 in allowFrames()
    valve->allowFrames(2);
    valve->step();

 
for (int i = 0; i < 2; i++)
{
    source->send(frames);
    valve->step();
}

BOOST_TEST(sinkQue->size() == 6);

//We are sending 4 frames again with enable false
for (int i = 0; i < 4; i++)
{
    source->send(frames);
    valve->step();
}

BOOST_TEST(sinkQue->size() == 6);

// The props are changed by passing 0 in allowFrames()
valve->allowFrames(0);
valve->step();


for (int i = 0; i < 2; i++)
{
    source->send(frames);
    valve->step();
}
}

BOOST_AUTO_TEST_CASE(multiple_pins)
{
    /*src - 2 output pins
    valve module - sieve = disabled
    2 children modules of valve */
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
    auto pinId = source->addOutputPin(metadata);
    auto pinId1 = source->addOutputPin(metadata1);

    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    // always use disable sieve with valve module 
    source->setNext(valve);

    auto sink1 = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    auto sink2 = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->addOutputPin(metadata);
    valve->addOutputPin(metadata1);
    valve->setNext(sink1);
    valve->setNext(sink2);

    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink1->init());
    BOOST_TEST(sink2->init());

    auto sink1Que = sink1->getQue();
    auto sink2Que = sink2->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));
    auto frame1 = source->makeFrame(1023, pinId1);
    frames.insert(make_pair(pinId1, frame1));
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 0);
    BOOST_TEST(sink2Que->size() == 0);

    valve->allowFrames(2);
    valve->step();

    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 2);
    BOOST_TEST(sink2Que->size() == 2);

    // check the number of frames in the frame container of the sinks
    while (sink1Que->size() != 0)
    { 
        frame_container sink1Frames = sink1Que->pop();
        sink1Frames = sink1Que->pop();
        BOOST_TEST((sink1Frames.size() == 2));
        bool flagGen = false;
        bool flagRaw = false;
        for (auto framePair = sink1Frames.begin(); framePair != sink1Frames.end(); framePair++)
        {
            auto metadata = (framePair->second)->getMetadata();
            if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
            {
                flagGen = true;
            }
            if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
            {
                flagRaw = true;
            }
        }
        BOOST_TEST(flagGen);
        BOOST_TEST(flagRaw);
    }
 
    while (sink2Que->size() != 0)
    {
        frame_container sink2Frames = sink2Que->pop();
        sink2Frames = sink2Que->pop();
        BOOST_TEST(sink2Frames.size() == 2);
        bool flagGen = false;
        bool flagRaw = false;
        for (auto framePair = sink2Frames.begin(); framePair != sink2Frames.end(); framePair++)
        {
            auto metadata = (framePair->second)->getMetadata();
            if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
            {
                flagGen = true;
            }
            if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
            {
                flagRaw = true;
            }
        }
        BOOST_TEST(flagGen);
        BOOST_TEST(flagRaw);
    }
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->addOutputPin(metadata);
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));

    // default props is 2
    valve->allowFrames(2);
    valve->step();

    // We are sending 4 frames with enable true so only two frames (moduleprops = 2) pass through the valve
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 2);

    //We are resetting the moduleprops to 4 and enable true 
    // rvw - getProps - change the value - sending in setProps
    auto currentProps = valve->getProps();
    currentProps.noOfFramesToCapture = 4;
    auto newValue = ValveModuleProps(currentProps.noOfFramesToCapture);
    valve->setProps(newValue);
    valve->step();
    valve->step();

    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }
 
    BOOST_TEST(sinkQue->size() == 6);

    //We are sending 4 frames again with enable false
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 6);

}

BOOST_AUTO_TEST_CASE(start_open)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(-1)));
    source->setNext(valve);
    auto sink = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->addOutputPin(metadata);
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));
 
    // We are sending 4 frames (moduleprops = -1) and the valve is open so all frames pass through the valve
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 4);
  
    //Closing the valve module
    valve->allowFrames(0);
    valve->step();

    for (int i = 0; i < 12; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 4);

}

BOOST_AUTO_TEST_CASE(valve_relay)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
    auto pinId = source->addOutputPin(metadata);
    auto pinId1 = source->addOutputPin(metadata1);

    auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);

    auto sink1 = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    auto sink2 = boost::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->addOutputPin(metadata);
    valve->addOutputPin(metadata1);
    valve->setNext(sink1, false); //Sink1 is closed
    valve->setNext(sink2); //Sink2 is open

    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink1->init());
    BOOST_TEST(sink2->init());

    auto sink1Que = sink1->getQue();
    auto sink2Que = sink2->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));
    auto frame1 = source->makeFrame(1023, pinId1);
    frames.insert(make_pair(pinId1, frame1));
    valve->allowFrames(2);
    valve->step();
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 0);
    BOOST_TEST(sink2Que->size() == 2);

    //Sink1 which was closed is opened using relay and Sink2 is closed using relay
    valve->relay(sink1, true);
    valve->step();
    valve->relay(sink2, false);
    valve->step();

    valve->allowFrames(2);
    valve->step();
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 2);
    BOOST_TEST(sink2Que->size() == 2);

    //Now both sinks are closed and no frames will pass to them
    valve->relay(sink1, false);
    valve->step();
    valve->allowFrames(2);
    valve->step();
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 2);
    BOOST_TEST(sink2Que->size() == 2);
}
BOOST_AUTO_TEST_SUITE_END()