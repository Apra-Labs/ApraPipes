#include "FaceDetectorXform.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/dnn.hpp"
#include "AIPExceptions.h"
#include "ApraFaceInfo.h"
#include "FaceDetectsInfo.h"

class FaceDetectorXform::Detail
{
public:
	Detail(FaceDetectorXformProps &_props) : mProps(_props)
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
	int mFrameType;
	cv::dnn::Net network;
	cv::Mat inputBlob;
	cv::Mat detection;
	const std::string FACE_DETECTION_CONFIGURATION = "./data/assets/deploy.prototxt";
	const std::string FACE_DETECTION_WEIGHTS = "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	ApraFaceInfo faceInfo;
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
	mDetail->network = cv::dnn::readNetFromCaffe(mDetail->FACE_DETECTION_CONFIGURATION, mDetail->FACE_DETECTION_WEIGHTS);
	if (mDetail->network.empty())
	{
		LOG_ERROR << "Failed to load network with the given settings. Please check the loaded parameters.";
		return false;
	}
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
	mDetail->inputBlob = cv::dnn::blobFromImage(mDetail->mInputImg, mDetail->mProps.scaleFactor, cv::Size(mDetail->mInputImg.cols, mDetail->mInputImg.rows),
												{104., 177.0, 123.0}, false, false);
	mDetail->network.setInput(mDetail->inputBlob, "data");

	mDetail->detection = mDetail->network.forward("detection_out");

	cv::Mat detectionMatrix(mDetail->detection.size[2], mDetail->detection.size[3], CV_32F, mDetail->detection.ptr<float>());

	for (int i = 0; i < detectionMatrix.rows; i++)
	{
		float confidence = detectionMatrix.at<float>(i, 2);

		if (confidence < mDetail->mProps.confidenceThreshold)
		{
			continue;
		}

		mDetail->faceInfo.x1 = detectionMatrix.at<float>(i, 3) * mDetail->mInputImg.cols;
		mDetail->faceInfo.y2 = detectionMatrix.at<float>(i, 4) * mDetail->mInputImg.rows;
		mDetail->faceInfo.x2 = detectionMatrix.at<float>(i, 5) * mDetail->mInputImg.cols;
		mDetail->faceInfo.y1 = detectionMatrix.at<float>(i, 6) * mDetail->mInputImg.rows;
		mDetail->faceInfo.score = confidence;

		mDetail->faces.emplace_back(mDetail->faceInfo);
	}

	mDetail->faceDetectsInfo.faces = mDetail->faces;
	auto outFrame = makeFrame(mDetail->faceDetectsInfo.getSerializeSize());
	mDetail->faceDetectsInfo.serialize(outFrame->data(), mDetail->faceDetectsInfo.getSerializeSize());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
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
