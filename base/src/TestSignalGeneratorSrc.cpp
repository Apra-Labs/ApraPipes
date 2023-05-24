#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include <cstdlib>
#include <cstdint>

class TestSignalGenerator::Detail
{
public:
     Detail(TestSignalGeneratorProps &_props) : mProps(_props), start_shade(0), end_shade(255), current_shade(start_shade){}
    

    ~Detail(){}
   
    bool generate(frame_sp &frame)
    {
        auto *frame_ptr = frame->data();
        uint8_t *x = static_cast<uint8_t *>(frame_ptr);

        for (int height = 0; height < mProps.height; height++)
        {
            memset(x, static_cast<uint8_t>(current_shade), mProps.width);
            x += mProps.width;
            current_shade += 1;
            if (current_shade > end_shade)
            {
                current_shade = start_shade;
            }
        }
        return true;
    }

    TestSignalGeneratorProps mProps;
    int start_shade;
    int end_shade;
    int current_shade;
};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
    : Module(SOURCE, "TestSignalGenerator", _props)
{
    mDetail.reset(new Detail(_props));
}

TestSignalGenerator::~TestSignalGenerator(){
    mDetail->~Detail();
};

bool TestSignalGenerator::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    framemetadata_sp metadata = getFirstOutputMetadata();
    auto frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool TestSignalGenerator::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool TestSignalGenerator::produce()
{
    size_t read_size = (getProps().width * getProps().height * 3) >> 1;
    auto mPinId = getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
    frame_container frames;
    frame_sp frame = makeFrame(read_size);
    mDetail->generate(frame);
    frames.insert(make_pair(mPinId, frame));
    send(frames);
    return true;
}

bool TestSignalGenerator::term()
{
    return Module::term();
}

void TestSignalGenerator::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}

TestSignalGeneratorProps TestSignalGenerator::getProps()
{
    return mDetail->mProps;
}

void TestSignalGenerator::setProps(TestSignalGeneratorProps& _props)
{
    mDetail->mProps = _props;
}


