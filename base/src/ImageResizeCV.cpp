#include "ImageResizeCV.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"


class ImageResizeCV::Detail
{
public:
	Detail(ImageResizeCVProps &_props) : mProps(_props)
	{
		outSize = cv::Size(mProps.width, mProps.height);
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}
	void setProps(ImageResizeCVProps &props)
	{
		mProps = props;
	}

public:

	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat iImg;
	cv::Mat oImg;
	cv::Size outSize;
	ImageResizeCVProps mProps;
};
ImageResizeCV::ImageResizeCV(ImageResizeCVProps _props) : Module(TRANSFORM, "ImageResizeCV", _props), mProps(_props), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props));
}

ImageResizeCV::~ImageResizeCV() {}

bool ImageResizeCV::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE )
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

bool ImageResizeCV::validateOutputPins()
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

void ImageResizeCV::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(mProps.width, mProps.height, rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}
std::string ImageResizeCV::addOutputPin(framemetadata_sp &metadata)
{
	mDetail->initMatImages(metadata);
	mDetail->mFrameLength = metadata->getDataSize();
	return Module::addOutputPin(metadata);
}
bool ImageResizeCV::init()
{
	return Module::init();
}

bool ImageResizeCV::term()
{
	return Module::term();
}

bool ImageResizeCV::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto outFrame = makeFrame();

	mDetail->iImg.data = static_cast<uint8_t *>(frame->data());
	mDetail->oImg.data = static_cast<uint8_t *>(outFrame->data());
	
	cv::resize(mDetail->iImg, mDetail->oImg, mDetail->outSize);
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	return true;
}

void ImageResizeCV::setMetadata(framemetadata_sp &metadata)
{
}
bool ImageResizeCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

ImageResizeCVProps ImageResizeCV::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

bool ImageResizeCV::handlePropsChange(frame_sp &frame)
{
	ImageResizeCVProps props(1.0, 0.0);
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void ImageResizeCV::setProps(ImageResizeCVProps &props)
{
	Module::addPropsToQueue(props);
}