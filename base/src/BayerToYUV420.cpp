#include "BayerToYUV420.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
 
class BayerToYUV420::Detail
{
public:
    Detail(BayerToYUV420Props &_props) : props(_props)
    {
    }
    ~Detail() {}
 
public:
    size_t mFrameLength;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    int currPixVal = 0;
 
private:
    BayerToYUV420Props props;
};
 
BayerToYUV420::BayerToYUV420(BayerToYUV420Props _props) : Module(TRANSFORM, "BayerToYUV420", _props), props(_props), mFrameType(FrameMetadata::GENERAL)
{
    mDetail.reset(new Detail(_props));
}
 
BayerToYUV420::~BayerToYUV420() {}
 
bool BayerToYUV420::validateInputPins()
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
 
bool BayerToYUV420::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    return true;
}
 
void BayerToYUV420::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}
 
std::string BayerToYUV420::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}
 
bool BayerToYUV420::init()
{
    return Module::init();
}
 
bool BayerToYUV420::term()
{
    return Module::term();
}
 
bool BayerToYUV420::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
    auto outFrame = makeFrame(WIDTH * HEIGHT * 1.5);
 
    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Frame is Empty";
        return true;
    }
 
    auto inpPtr = static_cast<uint16_t *>(frame->data());
    auto outPtr = static_cast<uint8_t *>(outFrame->data());
 
    memset(outPtr, 128, WIDTH * HEIGHT * 1.5); // remove for mono // yash earlier it was 128
 
    for (auto i = 0; i < HEIGHT; i++)
    {
        auto inpPtr1 = inpPtr + i * 800; // 800 Replace it with Width
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
 
void BayerToYUV420::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
 
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(WIDTH, HEIGHT, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
}
 
bool BayerToYUV420::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}
