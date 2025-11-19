#include "BrightnessContrastControlXform.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui.hpp"
#include "Utils.h"

class BrightnessContrastControl::Detail
{
public:
	Detail(BrightnessContrastControlProps &_props) : mProps(_props)
	{
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		mInputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}

	void setProps(BrightnessContrastControlProps &props)
	{
		mProps = props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat mInputImg;
	cv::Mat mOutputImg;
	BrightnessContrastControlProps mProps;
	int mFrameType;
};

BrightnessContrastControl::BrightnessContrastControl(BrightnessContrastControlProps _props) : Module(TRANSFORM, "BrightnessContrastControl", _props)
{
	mDetail.reset(new Detail(_props));
}

BrightnessContrastControl::~BrightnessContrastControl() {}

bool BrightnessContrastControl::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);

	if (rawMetadata->getDepth() != CV_8U)
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Bit depth not supported.");
	}

	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool BrightnessContrastControl::validateOutputPins()
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

void BrightnessContrastControl::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool BrightnessContrastControl::init()
{
	return Module::init();
}

bool BrightnessContrastControl::term()
{
	return Module::term();
}

bool BrightnessContrastControl::process(frame_container &frames)
{
	auto frame = frames.begin()->second;
	auto outFrame = makeFrame();

	mDetail->mInputImg.data = static_cast<uint8_t *>(frame->data());
	mDetail->mOutputImg.data = static_cast<uint8_t *>(outFrame->data());
	auto beta = std::round(mDetail->mProps.brightness * 255);
	mDetail->mInputImg.convertTo(mDetail->mOutputImg, -1, mDetail->mProps.contrast, beta);
	frames.insert({mDetail->mOutputPinId, outFrame});
	send(frames);
	return true;
}

void BrightnessContrastControl::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);

	RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);
	rawOutMetadata->setData(outputMetadata);
	auto imageType = rawMetadata->getImageType();

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
		throw AIPException(AIP_NOTIMPLEMENTED, "ImageType not Supported<" + std::to_string(imageType) + ">");
	}
}

bool BrightnessContrastControl::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

BrightnessContrastControlProps BrightnessContrastControl::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

bool BrightnessContrastControl::handlePropsChange(frame_sp &frame)
{
	BrightnessContrastControlProps props(1.0, 0.0);
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void BrightnessContrastControl::setProps(BrightnessContrastControlProps &props)
{
	Module::addPropsToQueue(props);
}
