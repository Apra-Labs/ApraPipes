#include "ImageOverlay.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include <opencv2/highgui/highgui.hpp>
#include "MetadataHints.h"
#include "DMAFDWrapper.h"

class ImageOverlay::Detail
{
public:
	Detail(ImageOverlayProps &_props) : mProps(_props)
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
	
	bool setOverlayMetadata(framemetadata_sp &metadata, frame_sp &frame) // remove frame unused
	{
		mOverlayedImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		mOverlayHeight = rawMetadata->getHeight();
		mOverlayWidth = rawMetadata->getWidth();
		setProps(mProps);
		return true;
	}

	bool setProps(ImageOverlayProps &_props)
	{
		// mRoi = cv::Rect(_props.xCoordinate * mImageWidth, _props.yCoordinate * mImageHeight, mOverlayedImg.cols, mOverlayedImg.rows);
		mRoi = cv::Rect(_props.topLeft * mImageWidth, _props.topRight * mImageHeight, mOverlayedImg.cols + (_props.topLeft * mImageWidth), mOverlayedImg.rows + (_props.topRight * mImageHeight));
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		if (!Utils::check_roi_bounds(mRoi, rawMetadata->getWidth(), rawMetadata->getHeight()))
		{

			LOG_ERROR << "ROI Out of Bound ,Using  Default Props";
			ImageOverlayProps defprops(0, 0, 0);
			mProps = defprops;
		}
		else
		{
			mProps = _props;
		}
		mRoi = cv::Rect(mProps.topLeft * mImageWidth, mProps.topRight * mImageHeight, mOverlayedImg.cols + (mProps.topLeft * mImageWidth), mOverlayedImg.rows + (mProps.topRight * mImageHeight));
		// cv::Mat roi2(src1, cv::Rect(0, 0, src2.cols, src2.rows));
	}

	void setOverlayFrame(frame_sp &frame)
	{
		overlayFrame = frame;
		flag = true;
	}

public:
	bool flag = false;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat mInputImg;
	cv::Mat mOutputImg;
	cv::Mat mOverlayedImg;
	cv::Mat mWeightedImage;
	cv::Mat mCroppedImage;
	long mOverlayWidth;
	long mOverlayHeight;
	int mImageWidth;
	int mImageHeight;
	ImageOverlayProps mProps;
	FrameMetadata::FrameType mFrameType;
	cv::Rect mRoi;
	frame_sp overlayFrame;
};

ImageOverlay::ImageOverlay(ImageOverlayProps _props) : Module(TRANSFORM, "ImageOverlay", _props)
{
	mDetail.reset(new Detail(_props));
}

ImageOverlay::~ImageOverlay() {}

bool ImageOverlay::validateInputPins()
{
	auto inputPinIdMetadataMap = getInputMetadata();
	for (auto const &element : inputPinIdMetadataMap)
	{
		auto &metadata = element.second;
		mDetail->mFrameType = metadata->getFrameType();
		if (mDetail->mFrameType != FrameMetadata::RAW_IMAGE)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << mDetail->mFrameType << ">";
			return false;
		}

		FrameMetadata::MemType memType = metadata->getMemType();
		if (memType != FrameMetadata::MemType::DMABUF)
		{
			LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
			return false;
		}
	}

	return true;
}

bool ImageOverlay::validateOutputPins()
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool ImageOverlay::validateInputOutputPins()
{
	if (getNumberOfInputPins() != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}
	return true;
}

void ImageOverlay::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	if (metadata->getHint() != OVERLAY_HINT)
	{
		if (mDetail->mFrameType == FrameMetadata::RAW_IMAGE)
		{
			mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::DMABUF));
		}
		mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
	}
	mDetail->mOutputMetadata->copyHint(*metadata.get());
}

bool ImageOverlay::init()
{
	return Module::init();
}

bool ImageOverlay::term()
{
	return Module::term();
}

bool ImageOverlay::process(frame_container &frames)
{
	frame_sp frame;
	int count = 0;
	for (auto const &element : frames)
	{
		count++;
	}

	for (auto const &element : frames)
	{
		auto tempFrame = element.second;
		auto outFrame = makeFrame(mDetail->mFrameLength);
		if (tempFrame->getMetadata()->getHint() == OVERLAY_HINT)
		{
			mDetail->setOverlayFrame(tempFrame);
			if (count == 1)
			{
				return true;
			}
		}
		else if(tempFrame->getMetadata()->getHint() == CAMERA_HINT)
		{
			frame = tempFrame;
			if(mDetail->flag == false)
			{
				frames.insert(make_pair(mDetail->mOutputPinId, frame));
				send(frames);
				return true;
			}
		}
	}
	
	mDetail->mInputImg.data = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(frame->data())->getHostPtr());
	mDetail->mOverlayedImg.data = static_cast<uint8_t *>(mDetail->overlayFrame->data());
	auto tf = mDetail->mInputImg;
	// mDetail->mCroppedImage = tf(mDetail->mRoi);
	// cv::addWeighted(mDetail->mCroppedImage, mDetail->mProps.alpha, mDetail->mOverlayedImg, (1 - mDetail->mProps.alpha), 0.0, mDetail->mWeightedImage);
	(mDetail->mWeightedImage).copyTo(mDetail->mInputImg(mDetail->mRoi));
	frames.insert(make_pair(mDetail->mOutputPinId, frame));
	send(frames);
	return true;
}

void ImageOverlay::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::DMABUF, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);
	rawOutMetadata->setData(outputMetadata);
	auto imageType = rawMetadata->getImageType();
	mDetail->mFrameLength = mDetail->mOutputMetadata->getDataSize();
	mDetail->initMatImages(metadata);
	mDetail->mImageWidth = rawMetadata->getWidth();
	mDetail->mImageHeight = rawMetadata->getHeight();
}

bool ImageOverlay::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() != mDetail->mFrameType)
	{
		throw AIPException(AIP_FATAL, "FrameType change not supported");
	}
	if (metadata->getHint() != OVERLAY_HINT)
	{
		setMetadata(metadata);
	}
	else
	{
		auto frame = makeFrame(metadata->getDataSize());
		mDetail->setOverlayMetadata(metadata, frame);
	}
	return true;
}

ImageOverlayProps ImageOverlay::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

void ImageOverlay::setProps(ImageOverlayProps &props)
{
	Module::addPropsToQueue(props);
}

bool ImageOverlay::handlePropsChange(frame_sp &frame)
{
	ImageOverlayProps props(0, 0, 0.0);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}
