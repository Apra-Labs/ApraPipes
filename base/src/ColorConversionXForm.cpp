#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "ColorConversionXForm.h"
#include "ColorConversionStrategy.h"
#include "AbsColorConversionFactory.h"

ColorConversion::ColorConversion(ColorConversionProps _props) : Module(TRANSFORM, "ColorConversion", _props), mProps(_props)
{
	mDetail.reset(new (DetailAbstract));
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
	return true;
}

void ColorConversion::setConversionStrategy(framemetadata_sp inputMetadata, framemetadata_sp outputMetadata)
{
	mDetail = AbsColorConversionFactory::create(inputMetadata, outputMetadata,mDetail->inpImg,mDetail->outImg);
}

void ColorConversion::addInputPin(framemetadata_sp& metadata, std::string_view pinId)
{
	mInputMetadata = metadata;
	Module::addInputPin(metadata, pinId);

	auto frameType = metadata->getFrameType();
	if (mProps.type == ColorConversionProps::RGB_TO_YUV420PLANAR)
	{
		mOutputMetadata = std::make_shared<RawImagePlanarMetadata>(FrameMetadata::HOST);
	}
	else
	{
		mOutputMetadata = std::make_shared<RawImageMetadata>(FrameMetadata::HOST);
	}
	mOutputPinId = addOutputPin(mOutputMetadata);
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
	frames.insert({mOutputPinId, outFrame});
	send(frames);
	return true;
}

bool ColorConversion::setMetadata(framemetadata_sp& metadata)
{
	int mWidth;
	int mHeight;
	auto inputFrameType = metadata->getFrameType();
	RawImageMetadata rawMetadata;
	RawImagePlanarMetadata rawPlanarMetadata(FrameMetadata::HOST);
	if (inputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto tempRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		mWidth = tempRawMetadata->getWidth();
		mHeight = tempRawMetadata->getHeight();
	}
	else if (inputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto tempRawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		mWidth = tempRawPlanarMetadata->getWidth(0);
		mHeight = tempRawPlanarMetadata->getHeight(0);
	}
	switch (mProps.type)
	{
	case ColorConversionProps::ConversionType::RGB_TO_MONO:
	case ColorConversionProps::ConversionType::BGR_TO_MONO:
	case ColorConversionProps::ConversionType::BAYERBG8_TO_MONO:
		rawMetadata = RawImageMetadata(mWidth, mHeight, ImageMetadata::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true);
		break;
	case ColorConversionProps::ConversionType::BGR_TO_RGB:
	case ColorConversionProps::ConversionType::BAYERBG8_TO_RGB:
	case ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB:
	case ColorConversionProps::ConversionType::BAYERGB8_TO_RGB:
	case ColorConversionProps::ConversionType::BAYERGR8_TO_RGB:
	case ColorConversionProps::ConversionType::BAYERRG8_TO_RGB:
		rawMetadata = RawImageMetadata(mWidth, mHeight, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
		break;
	case ColorConversionProps::ConversionType::RGB_TO_BGR:
		rawMetadata = RawImageMetadata(mWidth, mHeight, ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
		break;
	case ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR:
		rawPlanarMetadata = RawImagePlanarMetadata(mWidth, mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
		break;
	default:
		throw AIPException(AIP_FATAL, "conversion not supported");
	}

	if(mProps.type == ColorConversionProps::RGB_TO_YUV420PLANAR)
	{
		auto rawPlanarOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		rawPlanarOutMetadata->setData(rawPlanarMetadata);
	}
	else 
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(rawMetadata);
	}

	return true;
}

bool ColorConversion::processSOS(frame_sp& frame)
{
	auto mInputMetadata = frame->getMetadata();
	setMetadata(mInputMetadata);
	Module::setMetadata(mOutputPinId, mOutputMetadata);
	setConversionStrategy(mInputMetadata, mOutputMetadata);
	mShouldTriggerSos = false;
	return true;
}

bool ColorConversion::shouldTriggerSOS()
{
	return mShouldTriggerSos;
}