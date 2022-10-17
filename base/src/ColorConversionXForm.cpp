#include "FrameMetadata.h"
#include "ColorConversionXForm.h"
#include "ColorConversionStrategy.h"
#include "AbsColorConversionFactory.h"

ColorConversion::ColorConversion(ColorConversionProps _props) : Module(TRANSFORM, "ColorConversion", _props), mProps(_props), mFrameType(FrameMetadata::GENERAL)
{
}

ColorConversion::~ColorConversion()
{
}

bool ColorConversion::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}
	framemetadata_sp metadata = getFirstInputMetadata();
	auto frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image or Raw_Image_Planar. Actual<" << frameType << ">";
		return false;
	}
	if (frameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		auto imageType = rawMetadata->getImageType();
		if (imageType != ImageMetadata::RGB && imageType != ImageMetadata::BGR && imageType != ImageMetadata::BAYERBG10 && imageType != ImageMetadata::BAYERBG8 && imageType != ImageMetadata::BAYERGB8 && imageType != ImageMetadata::BAYERGR8 && imageType != ImageMetadata::BAYERRG8)
		{
			LOG_ERROR << "<" << getId() << ">Input Image type is not supported. Actual<" << imageType << ">";
			return false;
		}
	}
	else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		auto imageType = rawMetadata->getImageType();
		if (imageType != ImageMetadata::YUV420)
		{
			LOG_ERROR << "<" << getId() << ">Output Image type is not supported . Actual<" << imageType << ">";
			return false;
		}
	}
	return true;
}

bool ColorConversion::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	auto frameType = metadata->getFrameType();
	if ((frameType != FrameMetadata::RAW_IMAGE) && (frameType != FrameMetadata::RAW_IMAGE_PLANAR))
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or Raw_Image_Planar. Actual<" << frameType << ">";
		return false;
	}
	if (frameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		auto imageType = rawMetadata->getImageType();
		if (imageType != ImageMetadata::MONO && imageType != ImageMetadata::RGB && imageType != ImageMetadata::BGR && imageType != ImageMetadata::BAYERBG8)
		{
			LOG_ERROR << "<" << getId() << ">Output Image type is not supported . Actual<" << imageType << ">";
			return false;
		}
	}
	else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		auto imageType = rawMetadata->getImageType();
		if (imageType != ImageMetadata::YUV420)
		{
			LOG_ERROR << "<" << getId() << ">Output Image type is not supported . Actual<" << imageType << ">";
			return false;
		}
	}
	return true;
}

void ColorConversion::setConversionStrategy(framemetadata_sp inputMetadata, framemetadata_sp outputMetadata)
{
	mDetail = AbsColorConversionFactory::create(inputMetadata, outputMetadata);
}

void ColorConversion::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputFrameType = metadata->getFrameType();

	if (inputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		mWidth = rawMetadata->getWidth();
		mHeight = rawMetadata->getHeight();
	}
	else if (inputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		mWidth = rawPlanarMetadata->getWidth(0);
		mHeight = rawPlanarMetadata->getHeight(0);
	}

	switch (mProps.type)
	{
	case ColorConversionProps::ConversionType::RGB_2_MONO:
	case ColorConversionProps::ConversionType::BGR_2_MONO:
	case ColorConversionProps::ConversionType::BAYERBG8_2_MONO:
		mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(mWidth, mHeight, ImageMetadata::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
		break;
	case ColorConversionProps::ConversionType::BGR_2_RGB:
	case ColorConversionProps::ConversionType::BAYERBG8_2_RGB:
	case ColorConversionProps::ConversionType::YUV420PLANAR_2_RGB:
	case ColorConversionProps::ConversionType::BAYERGB8_2_RGB:
	case ColorConversionProps::ConversionType::BAYERGR8_2_RGB:
	case ColorConversionProps::ConversionType::BAYERRG8_2_RGB:
		mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(mWidth, mHeight, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
		break;
	case ColorConversionProps::ConversionType::RGB_2_BGR:
		mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(mWidth, mHeight, ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
		break;
	case ColorConversionProps::ConversionType::RGB_2_YUV420PLANAR:
		mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(mWidth, mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
		break;
	default:
		throw AIPException(AIP_FATAL, "conversion not supported");
	}
	mOutputPinId = addOutputPin(mOutputMetadata);
	setConversionStrategy(metadata, mOutputMetadata);
}

std::string ColorConversion::addOutputPin(framemetadata_sp& metadata)
{
	return Module::addOutputPin(metadata);
}

bool ColorConversion::init()
{
	return Module::init();
}

bool ColorConversion::term()
{
	return Module::term();
}

bool ColorConversion::process(frame_container& frames)
{
	auto outFrame = makeFrame();
	mDetail->convert(frames, outFrame, mOutputMetadata);
	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);
	return true;
}

bool ColorConversion::processSOS(frame_sp& frame)
{
	return true;
}