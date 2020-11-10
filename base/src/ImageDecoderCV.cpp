#include "ImageDecoderCV.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"

class ImageDecoderCV::Detail
{

public:
	Detail() {}

	~Detail() {}

	void setMetadata(framemetadata_sp& metadata) {
		mImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
		mMetadata = metadata;
	}

	framemetadata_sp getMetadata()
	{
		return mMetadata;
	}
	
	cv::Mat mImg;
private:
	
	framemetadata_sp mMetadata;
};

ImageDecoderCV::ImageDecoderCV(ImageDecoderCVProps _props) : Module(TRANSFORM, "ImageDecoderCV", _props)
{	
	mDetail.reset(new Detail());
}

ImageDecoderCV::~ImageDecoderCV() {}

bool ImageDecoderCV::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool ImageDecoderCV::validateOutputPins()
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

bool ImageDecoderCV::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstOutputMetadata();
	if (metadata->isSet())
	{
		mDetail->setMetadata(metadata);
	}

	return true;
}

bool ImageDecoderCV::term()
{
	return Module::term();
}

bool ImageDecoderCV::process(frame_container& frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::ENCODED_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto metadata = mDetail->getMetadata();
	auto outFrame = makeFrame(metadata->getDataSize(), metadata);

	mDetail->mImg.data = (uchar *) outFrame->data();
	cv::imdecode(cv::Mat(1, (int)frame->size(), CV_8UC1, frame->data()), cv::IMREAD_UNCHANGED, &mDetail->mImg);
	
	frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::RAW_IMAGE), outFrame));
	send(frames);
	return true;
}

bool ImageDecoderCV::processSOS(frame_sp& frame)
{
	cv::Mat matImg = cv::imdecode(cv::Mat(1, (int)frame->size(), CV_8UC1, frame->data()), cv::IMREAD_UNCHANGED);
	if (!matImg.data)
	{
		return false;
	}

	auto metadata = getFirstOutputMetadata();
	auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	rawImageMetadata->setData(matImg);
		
	mDetail->setMetadata(metadata);

	return true;
}