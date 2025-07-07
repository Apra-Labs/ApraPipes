#include "PerspectiveTransform.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"

class PerspectiveTransform::Detail
{
public:
    Detail(PerspectiveTransformProps &_props) : mProps(_props)
    {
        // Compute the transformation matrix from src and dst points
        transformMatrix = cv::getPerspectiveTransform(mProps.srcPoints, mProps.dstPoints);
    }
    ~Detail() {}

    void initMatImages(framemetadata_sp &input)
    {
        iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
        oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
    }

public:
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    cv::Mat iImg;
    cv::Mat oImg;
    cv::Mat transformMatrix;

private:
    PerspectiveTransformProps mProps;
};

PerspectiveTransform::PerspectiveTransform(PerspectiveTransformProps _props) : Module(TRANSFORM, "PerspectiveTransform", _props), mProps(_props), mFrameType(FrameMetadata::GENERAL)
{
    mDetail.reset(new Detail(_props));
}

PerspectiveTransform::~PerspectiveTransform() {}

bool PerspectiveTransform::validateInputPins()
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
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    auto imageType = rawMetadata->getImageType();
    switch (imageType)
    {
    case ImageMetadata::MONO:
    case ImageMetadata::BGR:
    case ImageMetadata::BGRA:
    case ImageMetadata::RGB:
    case ImageMetadata::RGBA:
        break;
    default:
        throw AIPException(AIP_NOTIMPLEMENTED, "Encoder not supported for ImageType<" + std::to_string(imageType) + ">");
    }
    return true;
}

bool PerspectiveTransform::validateOutputPins()
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

void PerspectiveTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true));
    mDetail->initMatImages(metadata);
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

std::string PerspectiveTransform::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}

bool PerspectiveTransform::init()
{
    return Module::init();
}

bool PerspectiveTransform::term()
{
    return Module::term();
}

bool PerspectiveTransform::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
    if (isFrameEmpty(frame))
    {
        return true;
    }

    auto outFrame = makeFrame();

    mDetail->iImg.data = static_cast<uint8_t *>(frame->data());
    mDetail->oImg.data = static_cast<uint8_t *>(outFrame->data());

    LOG_INFO << "Transformation Matrix: " << mDetail->transformMatrix;
    
    cv::warpPerspective(mDetail->iImg, mDetail->oImg, mDetail->transformMatrix, mDetail->oImg.size());
    frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
    send(frames);
    return true;
}

bool PerspectiveTransform::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    return true;
}