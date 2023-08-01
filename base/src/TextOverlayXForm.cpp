#include "TextOverlayXForm.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include <chrono>
#include <ctime>
#include <iomanip>

class TextOverlayXForm::Detail
{
public:
	Detail(TextOverlayXFormProps &_props) : mProps(_props)
	{
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		mInputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}

	void setProps(TextOverlayXFormProps &_props)
	{
		mProps = _props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	cv::Mat mInputImg;
	cv::Mat mOutputImg;
	int width;
	int height;
	TextOverlayXFormProps mProps;
};

TextOverlayXForm::TextOverlayXForm(TextOverlayXFormProps _props) : Module(TRANSFORM, "TextOverlayXForm", _props)
{
	mDetail.reset(new Detail(_props));
}

TextOverlayXForm::~TextOverlayXForm() {}

bool TextOverlayXForm::validateInputPins()
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

bool TextOverlayXForm::validateOutputPins()
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

void TextOverlayXForm::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool TextOverlayXForm::init()
{
	return Module::init();
}

bool TextOverlayXForm::term()
{
	return Module::term();
}

bool TextOverlayXForm::process(frame_container &frames)
{
	auto frame = frames.begin()->second;
	mDetail->mInputImg.data = static_cast<uint8_t *>(frame->data());

	int r, g, b, backr, backb, backg;
	sscanf(mDetail->mProps.fontColor.c_str(), "%02x%02x%02x", &r, &g, &b);

	auto outText = mDetail->mProps.text;
	if (mDetail->mProps.isDateTime)
	{
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
		outText = mDetail->mProps.text + " " + oss.str();
	}

	auto textSize = cv::getTextSize(outText, cv::FONT_HERSHEY_PLAIN, mDetail->mProps.fontSize / 10.0, 1, 0);
	int x, y;
	int padding = 10;
	if (mDetail->mProps.position == "UpperLeft")
	{
		x = padding;
		y = padding + textSize.height;
	}
	else if (mDetail->mProps.position == "LowerLeft")
	{
		x = padding;
		y = mDetail->height - padding;
	}
	else if (mDetail->mProps.position == "UpperRight")
	{
		x = mDetail->width - padding - textSize.width;
		y = padding + textSize.height;
	}
	else
	{
		x = mDetail->width - padding - textSize.width;
		y = mDetail->height - padding;
	}

	cv::Point point1, point2;

	if (mDetail->mProps.position == "UpperLeft" || mDetail->mProps.position == "UpperRight")
	{
		point1 = cv::Point(0, textSize.height + 2 * padding);
		point2 = cv::Point(mDetail->width, 0);
	}
	else
	{
		point1 = cv::Point(0, mDetail->height);
		point2 = cv::Point(mDetail->width, mDetail->height - textSize.height - 2 * padding);
	}

	sscanf(mDetail->mProps.backgroundColor.c_str(), "%02x%02x%02x", &backr, &backg, &backb);

	cv::Mat frameCopy = mDetail->mInputImg.clone();

	cv::rectangle(frameCopy,
				  point1,
				  point2,
				  cv::Scalar(backr, backg, backb),
				  -1);

	cv::putText(frameCopy,
				outText,
				cv::Point(x, y),
				cv::FONT_HERSHEY_PLAIN,
				mDetail->mProps.fontSize / 10.0,
				cv::Scalar(r, g, b),
				1,
				cv::LINE_AA);

	cv::addWeighted(frameCopy,
					mDetail->mProps.alpha,
					mDetail->mInputImg,
					1.0 - mDetail->mProps.alpha,
					0.0,
					mDetail->mInputImg);

	frames.insert(make_pair(mDetail->mOutputPinId, frame));
	send(frames);
	return true;
}

void TextOverlayXForm::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true);
	mDetail->width = rawMetadata->getWidth();
	mDetail->height = rawMetadata->getHeight(); 
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

TextOverlayXFormProps TextOverlayXForm::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}
void TextOverlayXForm::setProps(TextOverlayXFormProps &props)
{
	Module::addPropsToQueue(props);
}

bool TextOverlayXForm::handlePropsChange(frame_sp &frame)
{
	TextOverlayXFormProps props12(mDetail->mProps.alpha, mDetail->mProps.text, mDetail->mProps.position, mDetail->mProps.isDateTime, mDetail->mProps.fontSize, mDetail->mProps.fontColor, mDetail->mProps.backgroundColor);
	bool ret = Module::handlePropsChange(frame, props12);
	mDetail->setProps(props12);
	return ret;
}

bool TextOverlayXForm::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}
