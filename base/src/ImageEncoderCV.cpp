#include "ImageEncoderCV.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include<vector>


using namespace std;
class ImageEncoderCV::Detail
{

public:
	Detail(ImageEncoderCVProps &_props) 
	{
		flags.push_back(cv::IMWRITE_JPEG_QUALITY);
		flags.push_back(90);
	}

	~Detail(){}


	void setMetadata(framemetadata_sp &metadata)
	{
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));

		if (metadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE)
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			switch (rawImageMetadata->getImageType())
			{
			case ImageMetadata::ImageType::MONO:
				break;
			case ImageMetadata::ImageType::RGB:
				break;
			case ImageMetadata::ImageType::BGR:
				break;
			case ImageMetadata::ImageType::RGBA:
				break;
			case ImageMetadata::ImageType::BGRA:
				break;
			default:
				throw AIPException(AIP_NOTIMPLEMENTED, "Unknown imageType<" + std::to_string(rawImageMetadata->getImageType()) + ">");
			}
		}
		
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
		}

	}

	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat iImg;
	vector<int> flags;


private:
	ImageEncoderCVProps props;

};

ImageEncoderCV::ImageEncoderCV(ImageEncoderCVProps _props) : Module(TRANSFORM, "ImageEncoderCV", _props)
{	
	mDetail.reset(new Detail(_props));
	mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	mOutputPinId = addOutputPin(mOutputMetadata);
}

ImageEncoderCV::~ImageEncoderCV() {}

bool ImageEncoderCV::validateInputPins()
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool ImageEncoderCV::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}
bool ImageEncoderCV::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	if (metadata->isSet())
	{
		mDetail->setMetadata(metadata);
	}

	return true;

}

bool ImageEncoderCV::term()
{
	return Module::term();
}

bool ImageEncoderCV::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}
	vector<uchar> buf;
	
	mDetail->iImg.data = static_cast<uint8_t *>(frame->data());
	cv::imencode(".jpg",mDetail->iImg,buf,mDetail->flags);
	auto outFrame = makeFrame(buf.size(), mOutputMetadata);
	memcpy ( static_cast<void*>(outFrame->data()), &buf[0],buf.size());
	frames.insert(make_pair(mOutputPinId,outFrame));
	send(frames);
	return true;
}

bool ImageEncoderCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}
