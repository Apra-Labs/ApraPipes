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
	Detail(ImageResizeCVProps &_props) : props(_props)
	{
		outSize = cv::Size(props.width, props.height);
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}

public:
	
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat iImg;
	cv::Mat oImg;
	cv::Size outSize;

private:
	ImageResizeCVProps props;
};

ImageResizeCV::ImageResizeCV(ImageResizeCVProps _props) : Module(TRANSFORM, "ImageResizeCV", _props), props(_props), mFrameType(FrameMetadata::GENERAL)
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
	mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
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

	auto outFrame = makeFrame(mDetail->mFrameLength, mDetail->mOutputMetadata);

	mDetail->iImg.data = static_cast<uint8_t *>(frame->data());
	mDetail->oImg.data = static_cast<uint8_t *>(outFrame->data());

	cv::resize(mDetail->iImg, mDetail->oImg, mDetail->outSize);
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	return true;
}

void ImageResizeCV::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	RawImageMetadata outputMetadata(props.width, props.height, rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);//*****
	rawOutMetadata->setData(outputMetadata);
	auto imageType = rawMetadata->getImageType();

	mDetail->mFrameLength = mDetail->mOutputMetadata->getDataSize();
	mDetail->initMatImages(metadata);

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
bool ImageResizeCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}
