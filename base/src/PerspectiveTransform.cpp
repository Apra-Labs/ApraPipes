#include "PerspectiveTransform.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "ArrayMetadata.h"
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
        // Only compute transformation matrix in BASIC mode, for DYNAMIC mode, it will be computed per frame
        if (mProps.mode == PerspectiveTransformProps::BASIC)
        {
            transformMatrix = cv::getPerspectiveTransform(mProps.srcPoints, mProps.dstPoints);
        }
    }
    ~Detail() {}

    void initMatImages(framemetadata_sp &input)
    {
        iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
        oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
    }

    void updateTransformMatrix(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints)
    {
        transformMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
    }

public:
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    std::string mSrcPointsPinId;
    std::string mDstPointsPinId;
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
    if (mProps.mode == PerspectiveTransformProps::BASIC)
    {
        if (getNumberOfInputPins() != 1)
        {
            LOG_ERROR << "<" << getId() << ">::validateInputPins BASIC mode expects 1 input pin. Actual<" << getNumberOfInputPins() << ">";
            return false;
        }
    }
    else if (mProps.mode == PerspectiveTransformProps::DYNAMIC)
    {
        int numPins = getNumberOfInputPins();
        if (numPins < 1 || numPins > 3)
        {
            LOG_ERROR << "<" << getId() << ">::validateInputPins DYNAMIC mode expects 1-3 input pins during setup. Actual<" << numPins << ">";
            return false;
        }

        if (numPins > 0)
        {
            auto inputMetadata = getInputMetadata();
            int imageCount = 0, arrayCount = 0;
            
            for (auto const& elem: inputMetadata)
            {
                if (elem.second->getFrameType() == FrameMetadata::RAW_IMAGE)
                {
                    imageCount++;
                }
                else if (elem.second->getFrameType() == FrameMetadata::ARRAY)
                {
                    arrayCount++;
                }
            }
            
            if (imageCount > 1 || arrayCount > 2)
            {
                LOG_ERROR << "<" << getId() << ">::validateInputPins DYNAMIC mode expects at most 1 RAW_IMAGE and 2 ARRAY inputs. Actual: RAW_IMAGE<" << imageCount << "> ARRAY<" << arrayCount << ">";
                return false;
            }
        }
    }

    framemetadata_sp imageMetadata;
    if (mProps.mode == PerspectiveTransformProps::BASIC)
    {
        imageMetadata = getFirstInputMetadata();
    }
    else
    {
        auto inputMetadata = getInputMetadata();
        for (auto const& elem: inputMetadata)
        {
            if (elem.second->getFrameType() == FrameMetadata::RAW_IMAGE)
            {
                imageMetadata = elem.second;
                break;
            }
        }
    }

    // Only validate image metadata if we have it
    if (imageMetadata)
    {
        if (imageMetadata->getFrameType() != FrameMetadata::RAW_IMAGE)
        {
            LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image. Actual<" << imageMetadata->getFrameType() << ">";
            return false;
        }
        
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(imageMetadata);
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
    
    if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
    {
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
        mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true));
        mDetail->initMatImages(metadata);
        mDetail->mOutputMetadata->copyHint(*metadata.get());
        mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
    }
    else if (metadata->getFrameType() == FrameMetadata::ARRAY && mProps.mode == PerspectiveTransformProps::DYNAMIC)
    {
        if (mDetail->mSrcPointsPinId.empty())
        {
            mDetail->mSrcPointsPinId = pinId;
            LOG_INFO << "<" << getId() << "> Source points input pin: " << pinId;
        }
        else if (mDetail->mDstPointsPinId.empty())
        {
            mDetail->mDstPointsPinId = pinId;
            LOG_INFO << "<" << getId() << "> Destination points input pin: " << pinId;
        }
        else
        {
            LOG_ERROR << "<" << getId() << "> Too many ARRAY input pins. Expected 2 for DYNAMIC mode.";
        }
    }
}

std::string PerspectiveTransform::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}

bool PerspectiveTransform::init()
{
    if (mProps.mode == PerspectiveTransformProps::DYNAMIC)
    {
        if (getNumberOfInputPins() != 3)
        {
            LOG_ERROR << "<" << getId() << ">::init DYNAMIC mode requires exactly 3 input pins. Actual<" << getNumberOfInputPins() << ">";
            return false;
        }

        // Validate that we have exactly 1 RAW_IMAGE and 2 ARRAY inputs
        auto inputMetadata = getInputMetadata();
        int imageCount = 0, arrayCount = 0;
        
        for (auto const& elem: inputMetadata)
        {
            if (elem.second->getFrameType() == FrameMetadata::RAW_IMAGE)
            {
                imageCount++;
            }
            else if (elem.second->getFrameType() == FrameMetadata::ARRAY)
            {
                arrayCount++;
            }
        }
        
        if (imageCount != 1 || arrayCount != 2)
        {
            LOG_ERROR << "<" << getId() << ">::init DYNAMIC mode requires exactly 1 RAW_IMAGE and 2 ARRAY inputs. Actual: RAW_IMAGE<" << imageCount << "> ARRAY<" << arrayCount << ">";
            return false;
        }

        // Ensure we have identified both point pin IDs
        if (mDetail->mSrcPointsPinId.empty() || mDetail->mDstPointsPinId.empty())
        {
            LOG_ERROR << "<" << getId() << ">::init DYNAMIC mode requires both source and destination point pin IDs to be set";
            return false;
        }
    }

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

    // Handle DYNAMIC mode - extract points from input pins
    if (mProps.mode == PerspectiveTransformProps::DYNAMIC)
    {
        auto srcPointsFrame = frames.find(mDetail->mSrcPointsPinId);
        if (srcPointsFrame == frames.end() || isFrameEmpty(srcPointsFrame->second))
        {
            LOG_ERROR << "<" << getId() << "> Source points frame not found or empty in DYNAMIC mode";
            return true;
        }

        auto dstPointsFrame = frames.find(mDetail->mDstPointsPinId);
        if (dstPointsFrame == frames.end() || isFrameEmpty(dstPointsFrame->second))
        {
            LOG_ERROR << "<" << getId() << "> Destination points frame not found or empty in DYNAMIC mode";
            return true;
        }

        auto srcPointsData = static_cast<cv::Point2f*>(srcPointsFrame->second->data());
        auto dstPointsData = static_cast<cv::Point2f*>(dstPointsFrame->second->data());
        
        std::vector<cv::Point2f> srcPoints(srcPointsData, srcPointsData + 4);
        std::vector<cv::Point2f> dstPoints(dstPointsData, dstPointsData + 4);
        
        // Update transformation matrix with dynamic points
        mDetail->updateTransformMatrix(srcPoints, dstPoints);
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

void PerspectiveTransform::setProps(PerspectiveTransformProps &props)
{
    Module::addPropsToQueue(props);
}

PerspectiveTransformProps PerspectiveTransform::getProps()
{
    return mProps;
}

bool PerspectiveTransform::handlePropsChange(frame_sp &frame)
{
    PerspectiveTransformProps props;
    bool ret = Module::handlePropsChange(frame, props);
    
    // Update the transformation matrix if mode changed to BASIC or points changed
    if (props.mode == PerspectiveTransformProps::BASIC)
    {
        mDetail->updateTransformMatrix(props.srcPoints, props.dstPoints);
    }
    
    mProps = props;
    return ret;
}