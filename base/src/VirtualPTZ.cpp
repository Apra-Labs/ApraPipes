#include "VirtualPTZ.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include "AIPExceptions.h"

class VirtualPTZ::Detail
{
public:
    Detail(VirtualPTZProps &_props) : mProps(_props)
    {
    }
    ~Detail()
    {
    }

    void initMatImages(framemetadata_sp &input)
    {
        mInputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
        mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
    }

    void setProps(VirtualPTZProps &props)
    {
        if (!mOutputMetadata.get())
        {
            return;
        }
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
        mRoi = cv::Rect(mInputWidth * props.roiX, mInputHeight * props.roiY, mInputWidth * props.roiWidth, mInputHeight * props.roiHeight);
        if (!Utils::check_roi_bounds(mRoi, rawMetadata->getWidth(), rawMetadata->getHeight()))
        {
            LOG_ERROR << "Using the full image as roi is out of bounds. <" << props.roiX << "> <" << props.roiY << "> <" << props.roiWidth << "> <" << props.roiHeight << ">";
            VirtualPTZProps defProps(1, 1, 0, 0);
            mProps = defProps;
        }
        else
        {
            mProps = props;
        }
        mRoi = cv::Rect(mInputWidth * mProps.roiX, mInputHeight * mProps.roiY, mInputWidth * mProps.roiWidth, mInputHeight * mProps.roiHeight);
    }

public:
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    cv::Mat mOutputImg;
    cv::Mat mInputImg;
    cv::Size mOutSize;
    int mInputWidth;
    int mInputHeight;
    cv::Rect mRoi;
    VirtualPTZProps mProps;

private:
};

VirtualPTZ::VirtualPTZ(VirtualPTZProps _props) : Module(TRANSFORM, "VirtualPTZ", _props)
{
    mDetail.reset(new Detail(_props));
}

VirtualPTZ::~VirtualPTZ() {}

bool VirtualPTZ::validateInputPins()
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
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool VirtualPTZ::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

void VirtualPTZ::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool VirtualPTZ::init()
{
    return Module::init();
}

bool VirtualPTZ::term()
{
    return Module::term();
}

bool VirtualPTZ::process(frame_container &frames)
{
    auto frame = frames.begin()->second;
    auto outFrame = makeFrame();

    mDetail->mInputImg.data = static_cast<uint8_t *>(frame->data());
    mDetail->mOutputImg.data = static_cast<uint8_t *>(outFrame->data());

    cv::resize(mDetail->mInputImg(mDetail->mRoi), mDetail->mOutputImg, mDetail->mOutSize);
    frames.insert(make_pair(mDetail->mOutputPinId, outFrame));

    send(frames);
    return true;
}

void VirtualPTZ::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mDetail->mInputWidth = rawMetadata->getWidth();
    mDetail->mInputHeight = rawMetadata->getHeight();
    RawImageMetadata outputMetadata(mDetail->mInputWidth, mDetail->mInputHeight, rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true);
    auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);
    rawOutMetadata->setData(outputMetadata);
    mDetail->setProps(mDetail->mProps);
    auto imageType = rawMetadata->getImageType();
    mDetail->initMatImages(metadata);
    mDetail->mOutSize = cv::Size(mDetail->mInputWidth, mDetail->mInputHeight);

    switch (imageType)
    {
    case ImageMetadata::MONO:
    case ImageMetadata::BGR:
    case ImageMetadata::BGRA:
    case ImageMetadata::RGB:
    case ImageMetadata::RGBA:
        break;
    default:
        throw AIPException(AIP_NOTIMPLEMENTED, "ImageType Not Supported <" + std::to_string(imageType) + ">");
    }
}

void VirtualPTZ::setProps(VirtualPTZProps &props)
{
    Module::addPropsToQueue(props);
}

VirtualPTZProps VirtualPTZ::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

bool VirtualPTZ::handlePropsChange(frame_sp &frame)
{
    VirtualPTZProps props(0, 0, 0, 0);
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    return ret;
}

bool VirtualPTZ::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}
