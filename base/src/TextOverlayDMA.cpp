#include "TextOverlayDMA.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include <ctime>
#include "DMAFDWrapper.h"

class TextOverlayDMA::Detail
{
public:
	Detail(TextOverlayDMAProps &_props) : mProps(_props)
	{
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		mInputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}

	void setProps(TextOverlayDMAProps &_props)
	{
		// cv::Size textSize = cv::getTextSize(_props.text, cv::FONT_HERSHEY_COMPLEX_SMALL, 5, 2, &mBaseline);
		// if ((textSize.height + _props.yCoordinate > imageHeight) && (_props.xCoordinate + textSize.width > imageWidth))
		// 	LOG_ERROR << "Text is Out of Bound";
		// else
			mProps = _props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat mInputImg;
	cv::Mat mOutputImg;
	TextOverlayDMAProps mProps;
	int mBaseline;
	int imageHeight;
	int imageWidth;
};

TextOverlayDMA::TextOverlayDMA(TextOverlayDMAProps _props) : Module(TRANSFORM, "TextOverlayDMA", _props)
{
	mDetail.reset(new Detail(_props));
}

TextOverlayDMA::~TextOverlayDMA() {}

bool TextOverlayDMA::validateInputPins()
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

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool TextOverlayDMA::validateOutputPins()
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

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void TextOverlayDMA::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	// setMetadata(metadata);
	mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::DMABUF));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool TextOverlayDMA::init()
{
	return Module::init();
}

bool TextOverlayDMA::term()
{
	return Module::term();
}

bool TextOverlayDMA::process(frame_container &frames)
{
	auto frame = frames.begin()->second;
	// mDetail->mInputImg.data = static_cast<uint8_t *>(frame->data());

	mDetail->mInputImg.data = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(frame->data())->hostPtr);
	time_t now = time(0);  
  	char* dt = ctime(&now);
   	string newt = dt;
	// newt,
				// cv::Point(mDetail->mProps.xCoordinate, mDetail->mProps.yCoordinate),
	cv::putText(mDetail->mInputImg,
				"dyfgyfg",
				cv::Point(0,0),
				cv::FONT_HERSHEY_COMPLEX_SMALL,
				2.0,
				cv::Scalar(255, 255, 255),
				2,
				cv::LINE_AA);

	frames.insert(make_pair(mDetail->mOutputPinId, frame));
	send(frames);
	return true;
}

void TextOverlayDMA::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);																																
	RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::DMABUF, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata); 
	rawOutMetadata->setData(outputMetadata);
	auto imageType = rawMetadata->getImageType();
	mDetail->setProps(mDetail->mProps);
	mDetail->initMatImages(metadata);
	mDetail->imageHeight = rawMetadata->getHeight();
	mDetail->imageWidth = rawMetadata->getWidth();

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

TextOverlayDMAProps TextOverlayDMA::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}
void TextOverlayDMA::setProps(TextOverlayDMAProps &props)
{
	Module::addPropsToQueue(props);
}

bool TextOverlayDMA::handlePropsChange(frame_sp &frame)
{
	TextOverlayDMAProps props12(mDetail->mProps.text, mDetail->mProps.xCoordinate, mDetail->mProps.yCoordinate);
	bool ret = Module::handlePropsChange(frame, props12);
	mDetail->setProps(props12);
	return ret;
}

bool TextOverlayDMA::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);																																																																																																																																																																																																																																																																																																																																																																																																																																											(metadata);
	return true;
}