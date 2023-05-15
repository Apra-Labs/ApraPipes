#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include <cstdlib>
#include <cstdint>

class TestSignalGenerator::Detail
{
public:
    Detail(TestSignalGeneratorProps &_props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }
    void setProps(TestSignalGeneratorProps _props)
    {
        mProps = _props;
    }
    bool generate(frame_sp &frame)
    {
        auto *frame_ptr = frame->data();
        int start_shade = 0;
        int end_shade = 255;
        int steps = mProps.height / 3;
        int step = (end_shade - start_shade) / steps;
        int current_shade = start_shade;
        int *x = static_cast<int *>(frame_ptr);

        for (int height = 0; height < steps; height++)
        {
            // Loop of rows
            memset(x, (uint8_t)current_shade, mProps.width);
            x += mProps.width;
            // Update the shade value
            current_shade += step;
            if (current_shade > end_shade)
            {
                current_shade = start_shade;
            }
        }
        return true;
    }

public:
    TestSignalGeneratorProps mProps;
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
    size_t read_size = (mDetail->mProps.width * mDetail->mProps.height * 3) >> 1;
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