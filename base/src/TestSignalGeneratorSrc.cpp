#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include <cstdlib>
#include <cstdint>

class TestSignalGenerator::Detail
{
public:
    Detail(TestSignalGeneratorProps &_props)
        : mProps(_props), start_shade(0), end_shade(255), current_shade(start_shade) {}

    ~Detail() {}

    bool generate(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* x = static_cast<uint8_t*>(frame_ptr);

        for (int height = 0; height < mProps.height; height++)
        {
            memset(x, current_shade, mProps.width);
            x += mProps.width;
            current_shade += 1;
            if (current_shade > end_shade)
            {
                current_shade = start_shade;
            }
        }
        return true;
    }

    void setProps(const TestSignalGeneratorProps &_props)
    {
        mProps = _props;
        reset();
    }
    void reset()
    {
        current_shade = start_shade;
    }

    TestSignalGeneratorProps mProps;
    uint8_t start_shade = 0;
    uint8_t end_shade = 255;
    uint8_t current_shade = 0;
};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
    : Module(SOURCE, "TestSignalGenerator", _props), outputFrameSize(0)
{
    mDetail.reset(new Detail(_props));
    mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(_props.width, _props.height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    mOutputPinId = addOutputPin(mOutputMetadata);
}

TestSignalGenerator::~TestSignalGenerator()
{
    mDetail->~Detail();
}

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
        LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType should be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
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
    outputFrameSize = (getProps().width * getProps().height * 3) >> 1;

    return true;
}

bool TestSignalGenerator::produce()
{
    auto mPinId = getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
    frame_container frames;
    frame_sp frame = makeFrame(outputFrameSize);
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

bool TestSignalGenerator::handlePropsChange(frame_sp &frame)
{
    TestSignalGeneratorProps props;
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    outputFrameSize = (props.width * props.height * 3) >> 1;
    return ret;
}

void TestSignalGenerator::setProps(TestSignalGeneratorProps &props)
{
    Module::addPropsToQueue(props);
}

TestSignalGeneratorProps TestSignalGenerator::getProps()
{
    return mDetail->mProps;
}
