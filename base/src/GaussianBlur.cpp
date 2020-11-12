#include "GaussianBlur.h"
#include <opencv2/cudafilters.hpp>
#include "opencv2/core/cuda_stream_accessor.hpp"

class GaussianBlur::Detail
{
public:
    Detail(GaussianBlurProps &props): mProps(props)
    {        
        auto metadata = framemetadata_sp();
        setProps(props, metadata);
    }

    ~Detail()
    {
    }

    void setProps(GaussianBlurProps &props, framemetadata_sp& metadata)
    {
        mProps = props;
        if(!metadata.get())
        {
            return;
        }

        mStream = cv::cuda::StreamAccessor::wrapStream(props.stream->getCudaStream());

        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);

        mRoi = cv::Rect(props.roi[0], props.roi[1], props.roi[2], props.roi[3]);
        // validate roi
        if(!Utils::check_roi_bounds(mRoi, rawMetadata->getWidth(), rawMetadata->getHeight()))
        {
            LOG_ERROR << "Using the full image as roi is out of bounds. <" << props.roi[0] << "> <" << props.roi[1] << "> <" << props.roi[2] << "> <" << props.roi[3] << ">";
        } 
        // Utils::round_roi(mRoi, 512); // is it mandatory - test performance without this - this was added for visionworks functions on jetson
		mXOffset = rawMetadata->getOffset(mRoi.x, mRoi.y);
        mGaussianFilter = cv::cuda::createGaussianFilter(rawMetadata->getType(), rawMetadata->getType(), cv::Size(props.kernelSize, props.kernelSize), 1, 1);

        mSrc = Utils::getGPUMatHeader(mRoi, rawMetadata);
        mDst = Utils::getGPUMatHeader(mRoi, rawMetadata);
    }

    bool compute(frame_sp& inFrame, frame_sp& outFrame)
    {
        mSrc.data = static_cast<uchar*>(inFrame->data()) + mXOffset;
        mDst.data = static_cast<uchar*>(outFrame->data()) + mXOffset;
        mGaussianFilter->apply(mSrc, mDst, mStream);

		return true;
    }

    GaussianBlurProps mProps;    
private:
    cv::Ptr<cv::cuda::Filter> mGaussianFilter; // cv Ptr is a derived from std::shared_ptr
    cv::Rect mRoi;
    size_t mXOffset;
    cv::cuda::GpuMat mSrc, mDst;

    cv::cuda::Stream mStream;
};

GaussianBlur::GaussianBlur(GaussianBlurProps props) : Module(TRANSFORM, "GaussianBlur", props), mOutputPinId(""), mOutDataSize(0)
{
    mDetail.reset(new Detail(props));
}

GaussianBlur::~GaussianBlur()
{
}

bool GaussianBlur::validateInputPins()
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

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::CUDA_DEVICE)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
        return false;
    }

    return true;
}

bool GaussianBlur::validateOutputPins()
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

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::CUDA_DEVICE)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
        return false;
    }

    return true;
}

void GaussianBlur::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);

    mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
    mOutputMetadata->copyHint(*metadata.get());
    mOutputPinId = addOutputPin(mOutputMetadata);
}

bool GaussianBlur::init()
{
    if (!Module::init())
    {
        return false;
    }

    auto inputMetadata = getFirstInputMetadata();
    if (inputMetadata->isSet())
    {
		setMetadata(inputMetadata);
    }

    return true;
}

bool GaussianBlur::term()
{
    return Module::term();
}

bool GaussianBlur::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}
	
	auto outFrame = makeFrame(mOutDataSize, mOutputMetadata);

	if (!mDetail->compute(frame, outFrame))
	{
		return true;
	}

	frames.insert(make_pair(mOutputPinId, outFrame));

	send(frames);

    return true;
}

GaussianBlurProps GaussianBlur::getProps()
{
    fillProps(mDetail->mProps);

    return mDetail->mProps;
}

void GaussianBlur::setMetadata(framemetadata_sp& inputMetadata)
{
	mDetail->setProps(mDetail->mProps, inputMetadata);
	mOutDataSize = inputMetadata->getDataSize();
	auto rawInputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
	auto rawOutputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	rawOutputMetadata->setData(*rawInputMetadata);
}

bool GaussianBlur::processSOS(frame_sp &frame)
{
    auto inputMetadata = frame->getMetadata();
	setMetadata(inputMetadata);
    return true;
}

bool GaussianBlur::shouldTriggerSOS()
{
    return !mOutputMetadata->isSet();
}

bool GaussianBlur::processEOS(string &pinId)
{
    mOutputMetadata.reset();
    return true;
}

void GaussianBlur::setProps(GaussianBlurProps &props)
{
    Module::addPropsToQueue(props);
}

bool GaussianBlur::handlePropsChange(frame_sp &frame)
{
    GaussianBlurProps props(mDetail->mProps.stream);
    bool ret = Module::handlePropsChange(frame, props);

    mDetail->setProps(props, mOutputMetadata);

    return ret;
}