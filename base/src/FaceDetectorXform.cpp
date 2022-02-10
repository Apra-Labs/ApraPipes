#include "FaceDetectorXform.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "AIPExceptions.h"
#include "ApraFaceInfo.h"
#include "FaceDetectsInfo.h"

class FaceDetectorXform::Detail
{
public:
	Detail(FaceDetectorXformProps &_props) : mProps(_props), ultraface(binPath, paramPath, 320, 240, 1, 0.7)
	{
	}
	~Detail() {}

	void initMatImages(framemetadata_sp &input)
	{
		mInputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
	}

	void setProps(FaceDetectorXformProps &props)
	{
		mProps = props;
	}

public:
	framemetadata_sp mOutputMetadata;
	FaceDetectorXformProps mProps;
	std::string mOutputPinId;
	cv::Mat mInputImg;
	ncnn::Mat inmat;
	const std::string binPath = "../../data/version-RFB/RFB-320.bin";
	const std::string paramPath = "../../data/version-RFB/RFB-320.param";
	UltraFace ultraface;
	int mFrameType;
	ApraFaceInfo faceInfo;
	std::vector<FaceInfo> face_info;
	std::vector<ApraFaceInfo> faces;
	FaceDetectsInfo faceDetectsInfo;
};

FaceDetectorXform::FaceDetectorXform(FaceDetectorXformProps _props) : Module(TRANSFORM, "FaceDetectorXform", _props)
{
	mDetail.reset(new Detail(_props));
}

bool FaceDetectorXform::validateInputPins()
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool FaceDetectorXform::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}
	return true;
}

void FaceDetectorXform::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FACEDETECTS_INFO));
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool FaceDetectorXform::init()
{
	return Module::init();
}

bool FaceDetectorXform::term()
{
	return Module::term();
}

bool FaceDetectorXform::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	mDetail->mInputImg.data = static_cast<uint8_t *>(frame->data());
	mDetail->inmat = ncnn::Mat::from_pixels(mDetail->mInputImg.data, ncnn::Mat::PIXEL_BGR2RGB, mDetail->mInputImg.cols, mDetail->mInputImg.rows);

	mDetail->ultraface.detect(mDetail->inmat, mDetail->face_info);

	for (int i = 0; i < mDetail->face_info.size(); i++)
	{
		auto face = mDetail->face_info[i];
		mDetail->faceInfo.x1 = face.x1;
		mDetail->faceInfo.y1 = face.y1;
		mDetail->faceInfo.x2 = face.x2;
		mDetail->faceInfo.y2 = face.y2;
		mDetail->faceInfo.score = face.score;
		mDetail->faces.emplace_back(mDetail->faceInfo);
	}
	mDetail->faceDetectsInfo.faces = mDetail->faces;
	auto outFrame = makeFrame(mDetail->faceDetectsInfo.getSerializeSize());
	mDetail->faceDetectsInfo.serialize(outFrame->data(), mDetail->faceDetectsInfo.getSerializeSize());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	mDetail->face_info.clear();
	mDetail->faces.clear();
	mDetail->faceDetectsInfo.faces.clear();
	send(frames);
	return true;
}

void FaceDetectorXform::setMetadata(framemetadata_sp &metadata)
{
	if (!metadata->isSet())
	{
		return;
	}

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
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

bool FaceDetectorXform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

bool FaceDetectorXform::shouldTriggerSOS()
{
	if (mDetail->mInputImg.cols == 0)
	{
		return true;
	}
	return false;
}

FaceDetectorXformProps FaceDetectorXform::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

bool FaceDetectorXform::handlePropsChange(frame_sp &frame)
{
	FaceDetectorXformProps props;
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void FaceDetectorXform::setProps(FaceDetectorXformProps &props)
{
	Module::addPropsToQueue(props);
}
