#include "QRReader.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "RawImageMetadata.h"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

class QRReader::Detail
{

public:
	Detail(QRReaderProps _props) : mWidth(0), mHeight(0)
	{
		mReaderOptions.setEanAddOnSymbol(ZXing::EanAddOnSymbol::Read);
		LOG_INFO << "Setting tryHarder as " << _props.tryHarder;
		mReaderOptions.setTryHarder(_props.tryHarder);
		mSaveQRImages = _props.saveQRImages;
		mQRImagesFolderName = _props.qrImagesPath;
		mFrameRotationCounter = _props.frameRotationCounter;
		if(mFrameRotationCounter <= 0)
		{
			mFrameRotationCounter=1;
		}
		mFrameCounter = 0;
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
	ZXing::ReaderOptions mReaderOptions;
	std::string mOutputPinId;
	ZXing::ImageFormat mImageFormat;
	bool mSaveQRImages;
	fs::path mQRImagesFolderName;
	int mFrameCounter;
	int mFrameRotationCounter;
};

QRReader::QRReader(QRReaderProps _props) : Module(TRANSFORM, "QRReader", _props)
{
	mDetail.reset(new Detail(_props));
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
	boost::system::error_code ec;
	if (mDetail->mSaveQRImages && (!fs::create_directories(mDetail->mQRImagesFolderName, ec)))
	{
		if (ec)
		{
			LOG_ERROR << "Failed to create directory: " << mDetail->mQRImagesFolderName << ". Error: " << ec.message();
			mDetail->mQRImagesFolderName = "";
		}
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
 
	const auto &result = ZXing::ReadBarcode({static_cast<uint8_t *>(frame->data()), mDetail->mWidth, mDetail->mHeight, mDetail->mImageFormat}, mDetail->mReaderOptions);
	
	auto text = result.text();
	if(text.length())
	{
		LOG_INFO << "ZXING decoded QR: " << text;
	}
	
	if (mDetail->mSaveQRImages && (mDetail->mQRImagesFolderName != ""))
	{
		fs::path savePath = mDetail->mQRImagesFolderName / (std::to_string(mDetail->mFrameCounter) + ".raw");
        std::ofstream outFile(savePath.string(), std::ios::binary);
        if (outFile)
        {
            outFile.write(static_cast<char*>(frame->data()), frame->size());
            outFile.close();
        }
        else
        {
            LOG_ERROR << "Failed to save frame to " << savePath.string();
        }
		mDetail->mFrameCounter++;
		if ((mDetail->mFrameCounter % mDetail->mFrameRotationCounter) == 0)
		{
			mDetail->mFrameCounter = 0;
		}
	}
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