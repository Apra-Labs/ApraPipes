#include <stdafx.h>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <memory>
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
    std::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }

protected:
    bool process() { return false; }
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
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);
    auto sink = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert({pinId, frame});

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

    BOOST_TEST(sinkQue->size() == 4,"Four frames are passed to Sink");

    //The props are changed by passing 2 in allowFrames()
    valve->allowFrames(2);
    valve->step();


    for (int i = 0; i < 2; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 6,"Additional two frames are passed to Sink, total frames 6");

    //We are sending 4 frames again with enable false
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 6,"Enable false, so no frames were passed to sink, total frames remains 6");

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
    2 succesive modules of valve */
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadataGeneral = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto metadataRaw = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
    auto pinIdGeneral = source->addOutputPin(metadataGeneral);
    auto pinIdRaw = source->addOutputPin(metadataRaw);

    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    // always use disable sieve with valve module 
    source->setNext(valve);

    auto sink1 = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    auto sink2 = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));

    valve->setNext(sink1);
    valve->setNext(sink2);

    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink1->init());
    BOOST_TEST(sink2->init());

    auto sink1Que = sink1->getQue();
    auto sink2Que = sink2->getQue();

    auto frame = source->makeFrame(1023, pinIdGeneral);
    frame_container frames;
    frames.insert({pinIdGeneral, frame});
    auto frame1 = source->makeFrame(1023, pinIdRaw);
    frames.insert({pinIdRaw, frame1});
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 0,"Enable false so no frames are passed to sink1");
    BOOST_TEST(sink2Que->size() == 0,"Enable false so no frames are passed to sink2");

    valve->allowFrames(2);
    valve->step();

    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 2,"Two frames are passed to sink1");
    BOOST_TEST(sink2Que->size() == 2,"Two frames are passed to sink2");

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
        BOOST_TEST(flagGen,"Sink1 has frame of type General");
        BOOST_TEST(flagRaw,"Sink1 has frame of type Raw");
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
        BOOST_TEST(flagGen, "Sink2 has frame of type General");
        BOOST_TEST(flagRaw, "Sink2 has frame of type General");
    }
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);
    auto sink = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert({pinId, frame});

    // default props is 2
    valve->allowFrames(2);
    valve->step();

    // We are sending 4 frames with enable true so only two frames (moduleprops = 2) pass through the valve
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 2,"Two frames are passed due to props-2");

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

    BOOST_TEST(sinkQue->size() == 6,"Four frames are passed after changing props-4, total frames 6");

    //We are sending 4 frames again with enable false
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 6,"Zero frames are passed as props-0, total frames remains 6");

}

BOOST_AUTO_TEST_CASE(start_open)
{
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto pinId = source->addOutputPin(metadata);
    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(-1)));
    source->setNext(valve);
    auto sink = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto sinkQue = sink->getQue();

    auto frame = source->makeFrame(1023, pinId);
    frame_container frames;
    frames.insert({pinId, frame});

    // We are sending 4 frames (moduleprops = -1) and the valve is open so all frames pass through the valve
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 4,"Props=-1 so all frames are passed through Valve");

    //Closing the valve module
    valve->allowFrames(0);
    valve->step();

    for (int i = 0; i < 12; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sinkQue->size() == 4,"Props=0 so no additional frames are passed, total frames remains 4");

}

BOOST_AUTO_TEST_CASE(valve_relay)
{
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadataGeneral = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto metadataRaw = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
    auto pinIdGeneral = source->addOutputPin(metadataGeneral);
    auto pinIdRaw = source->addOutputPin(metadataRaw);

    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(2)));
    source->setNext(valve);

    auto sink1 = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    auto sink2 = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));

    valve->setNext(sink1, false); //Sink1 is closed
    valve->setNext(sink2); //Sink2 is open

    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink1->init());
    BOOST_TEST(sink2->init());

    auto sink1Que = sink1->getQue();
    auto sink2Que = sink2->getQue();

    auto frame = source->makeFrame(1023, pinIdGeneral);
    frame_container frames;
    frames.insert({pinIdGeneral, frame});
    auto frame1 = source->makeFrame(1023, pinIdRaw);
    frames.insert({pinIdRaw, frame1});
    valve->allowFrames(2);
    valve->step();
    for (int i = 0; i < 4; i++)
    {
        source->send(frames);
        valve->step();
    }

    BOOST_TEST(sink1Que->size() == 0,"No frames as Sink1 is closed");
    BOOST_TEST(sink2Que->size() == 2,"Two frames are passed to Sink2 which is open");

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

    BOOST_TEST(sink1Que->size() == 2, "Two frames are passed to Sink1 which is opened");
    BOOST_TEST(sink2Que->size() == 2, "No frames as Sink2 is now closed using relay, total frames remain 2");

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

    BOOST_TEST(sink1Que->size() == 2, "No frames as Sink1 is now closed using relay, total frames remain 2");
    BOOST_TEST(sink2Que->size() == 2, "No frames as Sink2 is still closed using relay, total frames remain 2");
}

BOOST_AUTO_TEST_CASE(multiType_frames)
{
    auto source = std::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadataGeneral = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
    auto metadataRaw = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
    auto pinIdGeneral = source->addOutputPin(metadataGeneral);
    auto pinIdRaw = source->addOutputPin(metadataRaw);

    auto valve = std::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(4)));
    source->setNext(valve);
    auto sink = std::shared_ptr<SinkModule>(new SinkModule(SinkModuleProps()));
    valve->setNext(sink);


    BOOST_TEST(source->init());
    BOOST_TEST(valve->init());
    BOOST_TEST(sink->init());

    auto frame = source->makeFrame(1023, pinIdGeneral);
    frame_container bothframes;
    bothframes.insert({pinIdGeneral, frame});
    auto frame1 = source->makeFrame(1023, pinIdRaw);
    bothframes.insert({pinIdRaw, frame1});

    auto gframe = source->makeFrame(1023, pinIdGeneral);
    frame_container generalframes;
    generalframes.insert({pinIdGeneral, gframe});

    auto rframe = source->makeFrame(1023, pinIdRaw);
    frame_container rawframes;
    rawframes.insert({pinIdRaw, rframe});

    valve->allowFrames(4);
    valve->step();

    auto sinkQue = sink->getQue();
    int framesGen = 0;
    int framesRaw = 0;

    source->send(generalframes);
    valve->step();
    frame_container sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
    }
    BOOST_TEST(framesGen == 1,"First frame container contains one frame of type General");
    BOOST_TEST(framesRaw == 0,"First frame container contains no frame of type Raw");

    source->send(generalframes);
    valve->step();
    sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
    }
    BOOST_TEST(framesGen == 2,"Second frame container contains one frame of type General, total gen frames = 2");
    BOOST_TEST(framesRaw == 0,"Second frame container contains no frame of type Raw, total raw frames = 0");

    source->send(bothframes);
    valve->step();
    sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
	}
    BOOST_TEST(framesGen == 3,"Third frame container contains one frame of type General, total gen frames = 3");
    BOOST_TEST(framesRaw == 1, "Third frame container contains one frame of type Raw, total raw frames = 1");

    source->send(bothframes);
    valve->step();
    sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
    }
    BOOST_TEST(framesGen == 4,"Fourth frame container contains one frame of type General, total gen frames = 4");
    BOOST_TEST(framesRaw == 2,"Fourth frame container contains one frame of type Raw, total raw frames = 2");

    source->send(rawframes);
    valve->step();
    sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
    }
    BOOST_TEST(framesGen == 4, "Fifth frame container contains no frame of type General, total gen frames = 4");
    BOOST_TEST(framesRaw == 3, "Fifth frame container contains one frame of type Raw, total raw frames = 3");

    source->send(rawframes);
    valve->step();
    sinkFrames = sinkQue->pop();
    for (auto framePair = sinkFrames.begin(); framePair != sinkFrames.end(); framePair++)
    {
        auto metadata = (framePair->second)->getMetadata();
        if ((metadata->getFrameType()) == FrameMetadata::GENERAL)
        {
            framesGen++;
        }
        if ((metadata->getFrameType()) == FrameMetadata::RAW_IMAGE)
        {
            framesRaw++;
        }
    }
    BOOST_TEST(framesGen == 4, "Sixth frame container contains no frame of type General, total gen frames = 4");
    BOOST_TEST(framesRaw == 4, "Sixth frame container contains one frame of type Raw, total raw frames = 4");

    source->send(bothframes);
    valve->step();

    BOOST_TEST(sinkQue->size() == 0,"All frame containers are popped for assertion so zero");
}

BOOST_AUTO_TEST_SUITE_END()