#include "QRReader.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "RawImageMetadata.h"

class QRReader::Detail
{

public:
	Detail() : mWidth(0), mHeight(0)
	{
		mHints.setEanAddOnSymbol(ZXing::EanAddOnSymbol::Read);
	}

	~Detail() {}

	void setMetadata(framemetadata_sp &metadata)
	{
		switch (metadata->getFrameType())
		{
		case FrameMetadata::FrameType::RAW_IMAGE:
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			if (rawImageMetadata->getImageType() != ImageMetadata::ImageType::RGB)
			{
				throw AIPException(AIP_FATAL, "Expected <RGB>. Actual <" + std::to_string(rawImageMetadata->getImageType()) + ">");
			}
			mWidth = rawImageMetadata->getWidth();
			mHeight = rawImageMetadata->getHeight();
			mImageFormat = ZXing::ImageFormat::RGB;
		}
		break;
		case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			if (rawImageMetadata->getImageType() != ImageMetadata::ImageType::YUV420)
			{
				throw AIPException(AIP_FATAL, "Expected <YUV420>. Actual <" + std::to_string(rawImageMetadata->getImageType()) + ">");
			}
			mWidth = rawImageMetadata->getWidth(0);
			mHeight = rawImageMetadata->getHeight(0);
			mImageFormat = ZXing::ImageFormat::Lum;
		}
		break;
		}
	}

	int mWidth;
	int mHeight;
	ZXing::DecodeHints mHints;
	std::string mOutputPinId;
	ZXing::ImageFormat mImageFormat;

private:
	framemetadata_sp mMetadata;
};

QRReader::QRReader(QRReaderProps _props) : Module(TRANSFORM, "QRReader", _props)
{
	mDetail.reset(new Detail());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	mDetail->mOutputPinId = addOutputPin(metadata);
}

QRReader::~QRReader() {}

bool QRReader::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool QRReader::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::GENERAL)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType is expected to be GENERAL. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool QRReader::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool QRReader::term()
{
	return Module::term();
}

bool QRReader::process(frame_container &frames)
{
	auto frame = frames.begin()->second;

	const auto &result = ZXing::ReadBarcode({static_cast<uint8_t *>(frame->data()), mDetail->mWidth, mDetail->mHeight, mDetail->mImageFormat}, mDetail->mHints);
	
	auto text = result.text();
	
	auto outFrame = makeFrame(text.length(), mDetail->mOutputPinId);
	memcpy(outFrame->data(), text.c_str(), outFrame->size());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));

	send(frames);

	return true;
}

bool QRReader::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool QRReader::shouldTriggerSOS()
{
	return mDetail->mWidth == 0;
}