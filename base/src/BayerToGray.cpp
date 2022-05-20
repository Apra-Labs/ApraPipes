#include "BayerToGray.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"

class BayerToGray::Detail
{
public:
    Detail(BayerToGrayProps &_props) : props(_props)
    {
    }
    ~Detail() {}

public:
    size_t mFrameLength;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;

private:
    BayerToGrayProps props;
};

BayerToGray::BayerToGray(BayerToGrayProps _props) : Module(TRANSFORM, "BayerToGray", _props), props(_props), mFrameType(FrameMetadata::GENERAL)
{
    mDetail.reset(new Detail(_props));
}

BayerToGray::~BayerToGray() {}

bool BayerToGray::validateInputPins()
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

bool BayerToGray::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    return true;
}

void BayerToGray::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(1024, 800, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

std::string BayerToGray::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}

bool BayerToGray::init()
{
    return Module::init();
}

bool BayerToGray::term()
{
    return Module::term();
}

bool BayerToGray::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);

    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Frame is Empty";
        return true;
    }

    auto outFrame = makeFrame();

    int width = 1024;
    int height = 800;
    auto inpPtr = static_cast<uint16_t *>(frame->data());
    auto outPtr = static_cast<uint8_t *>(outFrame->data());

    memset(outPtr, 0, 1024 * 800);

    for (auto i = 0; i < 800; i++)
    {
        auto inpPtr1 = inpPtr + i * 800;
        auto outPtr1 = outPtr + i * 1024;
        for (auto j = 0; j < 800; j++)
        {
            *outPtr1++ = (*inpPtr1++) >> 2;
        }
    }

    frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
    send(frames);
    return true;
}

void BayerToGray::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(1024, 800, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
}

bool BayerToGray::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}
