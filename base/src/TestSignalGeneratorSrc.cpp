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
    bool generate(frame_sp frame)
    {
        auto* frame_ptr=frame.get();
        void* ptr = static_cast<void*>(frame_ptr);
        int start_shade = 0;
        int end_shade = 255;
        int steps = 120;
        int step = (end_shade - start_shade) / steps;
        int current_shade = start_shade;

        for (int height = 0; height < 120; height++)
        {
            // Loop of rows
            memset(ptr, (uint8_t)current_shade, 640);
            ptr+=640;
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
    TestSignalGeneratorProps mProps ;
};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
    : Module(SOURCE, "TestSignalGenerator", _props)
{
    mDetail.reset(new Detail(_props));
}

TestSignalGenerator::~TestSignalGenerator(){};

bool TestSignalGenerator::validateOutputPins()
{
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