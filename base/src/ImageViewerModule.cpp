#include "stdafx.h"
#include "ImageViewerModule.h"
#include "Frame.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FrameMetadata.h"
#include "Logger.h"
#include "Utils.h"

class ImageViewerModule::Detail
{

public:
	Detail(std::string& strTitle): mStrTitle(strTitle) {}

	~Detail() {}

	void setMatImg(RawImageMetadata* rawMetadata)
	{
		mImg = Utils::getMatHeader(rawMetadata);
	}

	void showImage(frame_sp& frame)
	{
		mImg.data = (uchar *)frame->data();
		cv::imshow(mStrTitle, mImg);
		cv::waitKey(1); //use 33 for linux Grrr
	}

	bool shouldTriggerSOS()
	{
		return !mImg.rows;
	}
		
private:
	cv::Mat mImg;	
	std::string mStrTitle;
};

ImageViewerModule::ImageViewerModule(ImageViewerModuleProps _props) : Module(SINK, "ImageViewerModule", _props) {
	mDetail.reset(new Detail(_props.strTitle));
}

ImageViewerModule::~ImageViewerModule() {}

bool ImageViewerModule::validateInputPins()
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool ImageViewerModule::init() 
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	if (metadata->isSet())
	{
		mDetail->setMatImg(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
	}
		
	return true; 
}

bool ImageViewerModule::term() { return Module::term(); }

bool ImageViewerModule::process(frame_container& frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}
	
	mDetail->showImage(frame);
	return true;
}

bool ImageViewerModule::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMatImg(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
	return true;
}

bool ImageViewerModule::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}
