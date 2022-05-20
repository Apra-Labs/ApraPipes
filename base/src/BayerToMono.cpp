#include "BayerToMono.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#define WIDTH 800
#define HEIGHT 800

class BayerToMono::Detail
{
public:
    Detail(BayerToMonoProps &_props) : props(_props)
    {
    }
    ~Detail() {}

public:
    size_t mFrameLength;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    int currPixVal = 0;

private:
    BayerToMonoProps props;
};

BayerToMono::BayerToMono(BayerToMonoProps _props) : Module(TRANSFORM, "BayerToMono", _props), props(_props), mFrameType(FrameMetadata::GENERAL)
{
    mDetail.reset(new Detail(_props));
}

BayerToMono::~BayerToMono() {}

bool BayerToMono::validateInputPins()
{
    if (getNumberOfInputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstInputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool BayerToMono::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    return true;
}

void BayerToMono::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(WIDTH, HEIGHT, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

std::string BayerToMono::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}

bool BayerToMono::init()
{
    return Module::init();
}

bool BayerToMono::term()
{
    return Module::term();
}

bool BayerToMono::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
    auto outFrame = makeFrame(WIDTH * HEIGHT);

    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Frame is Empty";
        return true;
    }

    auto inpPtr = static_cast<uint16_t *>(frame->data());
    auto outPtr = static_cast<uint8_t *>(outFrame->data());

    for (auto i = 0; i < HEIGHT; i++)
    {
        auto inpPtr1 = inpPtr + i * 800;
        auto outPtr1 = outPtr + i * WIDTH;
        for (auto j = 0; j < WIDTH; j++)
        {
            *outPtr1++ = (*inpPtr1++) >> 2;
        }
    }
    frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
    send(frames);
    return true;
}

void BayerToMono::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(WIDTH, HEIGHT, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
}

bool BayerToMono::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}
