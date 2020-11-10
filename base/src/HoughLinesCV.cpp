#include "HoughLinesCV.h"
#include <cuda_runtime.h>
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "ROIMetadata.h"
#include "ApraLines.h"

#include "ApraHoughSegmentDetectorImpl.h"

// #include "opencv2/cudaimgproc.hpp"

class HoughLinesCV::Detail
{
public:
    Detail(HoughLinesCVProps &props, buffer_sp& linesBuffer) : mProps(props)
    {
        mLinesBuffer = linesBuffer;
        setProps(props);
    }

    ~Detail()
    {
    }

	void setProps(HoughLinesCVProps &props)
	{
		setProps(props, mMetadata);
	}

    void setProps(HoughLinesCVProps &props, framemetadata_sp &metadata)
    {
        mProps = props;
        if (!metadata.get())
        {
            return;
        }

        mMetadata = metadata;
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);

        mRoi = cv::Rect(props.roi[0], props.roi[1], props.roi[2], props.roi[3]);
        updateROI();

        mLines = cv::cuda::GpuMat(1, max_hough_lines, CV_32SC1, mLinesBuffer->data());

        // mOpenCVDetector = cv::cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), props.minLineLength, props.maxLineGap, max_hough_lines);

        mStream = props.stream->getCudaStream();
        mCVStream = cv::cuda::StreamAccessor::wrapStream(mStream);
    }

    frame_sp compute(frame_sp &threshold, frame_sp &roiFrame, std::function<frame_sp (size_t size)> makeFrame)
    {
        auto outFrame = makeFrame(0);

        if (roiFrame.get())
        {
            mRoi = ROIMetadata::deserialize(roiFrame->data());
            if (!ROIMetadata::isValid(mRoi))
            {
                // if all 0s then it is not a valid roi
                return outFrame;
            }
            updateROI();
        }

        mSrc.data = static_cast<uchar*>(threshold->data()) + mXOffset;

		// mOpenCVDetector->detect(mSrc, mLines);
		// auto noOfLines = mLines.cols;

        auto noOfLines = mDetector->detect(mSrc, mLines, mCVStream);
        if (noOfLines)
        {
            outFrame = makeFrame(noOfLines*4*4);            
            cudaMemcpyAsync(outFrame->data(), mLines.data, outFrame->size(), cudaMemcpyDeviceToHost, mStream);
            cudaStreamSynchronize(mStream);

            ApraLines lines(outFrame->data(), outFrame->size());
            for(auto i = 0; i < noOfLines; i++)
            {
                cv::Vec4i& line = lines[i];
                line[0] += mRoi.x;
                line[1] += mRoi.y;
                line[2] += mRoi.x;
                line[3] += mRoi.y;
            }
        }

        return outFrame;
    }

    HoughLinesCVProps mProps;
    static const int max_hough_lines = 1000;
    static const size_t hough_lines_buffer_size = max_hough_lines*4;

	framemetadata_sp mMetadata;

private:
    void updateROI()
    {
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mMetadata);

        // validate roi
        if (!Utils::check_roi_bounds(mRoi, rawMetadata->getWidth(), rawMetadata->getHeight()))
        {
            LOG_ERROR << "Using the full image as roi is out of bounds. <" << mProps.roi[0] << "> <" << mProps.roi[1] << "> <" << mProps.roi[2] << "> <" << mProps.roi[3] << ">";
        }

		Utils::round_roi(mRoi, 512); 
        mXOffset = rawMetadata->getOffset(mRoi.x, mRoi.y);
		mSrc = Utils::getGPUMatHeader(mRoi, rawMetadata);


		mDetector.reset();
        mDetector = std::auto_ptr<ApraHoughSegmentDetectorImpl>(new ApraHoughSegmentDetectorImpl(1.0f, (float)(CV_PI / 180.0f), mProps.minLineLength, mProps.maxLineGap, max_hough_lines));
        mDetector->init(mRoi.height, mRoi.width, rawMetadata->getType());
    }

    cv::Rect mRoi;
    size_t mXOffset;

	// cv::Ptr<cv::cuda::HoughSegmentDetector> mOpenCVDetector;

    std::auto_ptr<ApraHoughSegmentDetectorImpl> mDetector;
    cv::cuda::GpuMat mSrc, mLines;
    buffer_sp mLinesBuffer;

    cv::cuda::Stream mCVStream;
    cudaStream_t mStream;
};

HoughLinesCV::HoughLinesCV(HoughLinesCVProps props) : Module(TRANSFORM, "HoughLinesCV", props), mOutputPinId("")
{
    auto linesBuffer = makeBuffer(Detail::hough_lines_buffer_size, FrameMetadata::MemType::CUDA_DEVICE);
    mDetail.reset(new Detail(props, linesBuffer));
}

HoughLinesCV::~HoughLinesCV()
{
}

bool HoughLinesCV::validateInputPins()
{
    if (getNumberOfInputPins() > 2)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1 or 2. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    auto inputMetadataByPin = getInputMetadata();
    for (auto const &itr : inputMetadataByPin)
    {
        auto &metadata = itr.second;
        FrameMetadata::FrameType frameType = metadata->getFrameType();

        if (frameType == FrameMetadata::ROI)
        {
            continue;
        }

        if (frameType != FrameMetadata::RAW_IMAGE)
        {
            LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or ROI. Actual<" << frameType << ">";
            return false;
        }

        FrameMetadata::MemType memType = metadata->getMemType();
        if (memType != FrameMetadata::MemType::CUDA_DEVICE)
        {
            LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
            return false;
        }        
    }

    return true;
}

bool HoughLinesCV::validateInputOutputPins()
{
    if (getNumberOfInputsByType(FrameMetadata::RAW_IMAGE) != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputOutputPins 1 and only 1 RAW_IMAGE is mandatory";
        return false;
    }

    return Module::validateInputOutputPins();
}

bool HoughLinesCV::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::APRA_LINES)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be APRA_LINES. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

void HoughLinesCV::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);

    if (metadata->getFrameType() != FrameMetadata::FrameType::RAW_IMAGE)
    {
        return;
    }

    mOutputMetadata = framemetadata_sp(new ApraLinesMetadata());
    mOutputMetadata->copyHint(*metadata.get());
    mOutputPinId = addOutputPin(mOutputMetadata);
}

bool HoughLinesCV::init()
{
    if (!Module::init())
    {
        return false;
    }

    auto inputMetadata = getInputMetadataByType(FrameMetadata::FrameType::RAW_IMAGE);
    if (inputMetadata->isSet())
    {
        setMetadata(inputMetadata);
    }

    return true;
}

bool HoughLinesCV::term()
{
    return Module::term();
}

bool HoughLinesCV::process(frame_container &frames)
{
    frame_sp thresholdFrame = getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
    if(isFrameEmpty(thresholdFrame))
    {
        return true;
    }

    frame_sp roiFrame  = getFrameByType(frames, FrameMetadata::FrameType::ROI);
    frame_sp outFrame = mDetail->compute(thresholdFrame, roiFrame, [&](size_t size) {
        return makeFrame(size, mOutputMetadata);    
    });
    frames.insert(make_pair(mOutputPinId, outFrame));

    send(frames);

    return true;
}

HoughLinesCVProps HoughLinesCV::getProps()
{
    fillProps(mDetail->mProps);

    return mDetail->mProps;
}

void HoughLinesCV::setMetadata(framemetadata_sp &inputMetadata)
{
    mDetail->setProps(mDetail->mProps, inputMetadata);
    
    auto apraLinesMetadata = FrameMetadataFactory::downcast<ApraLinesMetadata>(mOutputMetadata);
    apraLinesMetadata->setParentMetadata(inputMetadata);
}

bool HoughLinesCV::processSOS(frame_sp &frame)
{
    auto inputMetadata = frame->getMetadata();
    if (inputMetadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE)
    {
        setMetadata(inputMetadata);
    }
    return true;
}

bool HoughLinesCV::shouldTriggerSOS()
{
    return !mDetail->mMetadata.get() || !mDetail->mMetadata->isSet();
}

bool HoughLinesCV::processEOS(string &pinId)
{   
    return true;
}

void HoughLinesCV::setProps(HoughLinesCVProps &props)
{
    Module::setProps(props, PropsChangeMetadata::ModuleName::HoughLinesCV);
}

bool HoughLinesCV::handlePropsChange(frame_sp &frame)
{
    HoughLinesCVProps props(mDetail->mProps.stream);
    bool ret = Module::handlePropsChange(frame, props);

    mDetail->setProps(props);

    return ret;
}