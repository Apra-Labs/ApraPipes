#include <boost/serialization/vector.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include "FacialLandmarksCV.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "ApraPoint2f.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "Overlay.h"

class Detail
{
public:
	Detail(FacialLandmarkCVProps &_props) : props(_props), mFrameType(FrameMetadata::GENERAL), mFrameLength(0) {}

	~Detail() {}

	virtual bool compute(frame_sp buffer) = 0;

	void initMatImages(framemetadata_sp& input)
	{
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
	}

	void setMetadata(framemetadata_sp& metadata)
	{
		mInputMetadata = metadata;
		if (mFrameType != metadata->getFrameType())
		{
			mFrameType = metadata->getFrameType();
			switch (mFrameType)
			{
			case FrameMetadata::RAW_IMAGE:
				mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::HOST));
				break;
			default:
				throw AIPException(AIP_FATAL, "Unsupported frameType<" + std::to_string(mFrameType) + ">");
			}
		}

		if (!metadata->isSet())
		{
			return;
		}

		ImageMetadata::ImageType imageType;
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		height = rawMetadata->getHeight();
		width = rawMetadata->getWidth();
		type = rawMetadata->getType();
		step = rawMetadata->getStep();
		RawImageMetadata outputMetadata(width, height, rawMetadata->getImageType(), rawMetadata->getType(), rawMetadata->getStep(), rawMetadata->getDepth(), FrameMetadata::HOST, true);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(outputMetadata);

		imageType = rawMetadata->getImageType();
		depth = rawMetadata->getDepth();

		switch (imageType)
		{
		case ImageMetadata::MONO:
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
		case ImageMetadata::BGRA:
		case ImageMetadata::RGBA:
			break;
		default:
			throw AIPException(AIP_NOTIMPLEMENTED, "ImageType not Supported<" + std::to_string(imageType) + ">");
		}

		mFrameLength = mOutputMetadata->getDataSize();
	}

	void setProps(FacialLandmarkCVProps& mprops)
	{
		setMetadata(mInputMetadata);
	}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	FacialLandmarkCVProps props;
	cv::Mat iImg;
	vector<vector<cv::Point2f>>landmarks;
	vector<cv::Rect> faces;

protected:
	framemetadata_sp mInputMetadata;
	FrameMetadata::FrameType mFrameType;
	uint32_t width, height, type, depth, step;
	
};

class DetailSSD : public Detail
{
public:
	DetailSSD(FacialLandmarkCVProps& _props) : Detail(_props)
	{
		cv::String modelConfiguration = props.Face_Detection_Configuration;
		cv::String modelBinary = props.Face_Detection_Weights;

		facemark = props.facemark;
		facemark->loadModel(props.landmarksDetectionModel);

	    faceDetector = cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary);
	}
	
	bool compute(frame_sp buffer)
	{
		//input must be 3 channel image(RGB)
	    // Create a 4-dimensional blob from the image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
        iImg.data = static_cast<uint8_t *>(buffer->data());
		cv::Mat inputBlob = cv::dnn::blobFromImage(iImg, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);

		// Set the input blob as input to the face detector network
		faceDetector.setInput(inputBlob, "data");

		// Forward propagate the input through the network and obtain the output
		cv::Mat detection = faceDetector.forward("detection_out");

		cv::Mat detectionMatrix(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		for (int i = 0; i < detectionMatrix.rows; i++)
		{
			float confidence = detectionMatrix.at<float>(i, 2);

			if (confidence > 0.5) // Set the confidence threshold for face detection
			{
				int x1 = static_cast<int>(detectionMatrix.at<float>(i, 3) * iImg.cols);
				int y1 = static_cast<int>(detectionMatrix.at<float>(i, 4) * iImg.rows);
				int x2 = static_cast<int>(detectionMatrix.at<float>(i, 5) * iImg.cols);
				int y2 = static_cast<int>(detectionMatrix.at<float>(i, 6) * iImg.rows);

				cv::Rect faceRect(x1, y1, x2 - x1, y2 - y1);

				faces.push_back(faceRect);
			}
		}

		if (faces.size() == 0) {
			return false;
		}

		bool success = facemark->fit(iImg, faces, landmarks);

		return success;
	}

private:
	cv::dnn::Net faceDetector;
	cv::Ptr<cv::face::Facemark> facemark;
};

class DetailHCASCADE : public Detail
{
public:
	DetailHCASCADE(FacialLandmarkCVProps& _props) : Detail(_props), faceDetector(props.faceDetectionModel)
	{
		facemark = props.facemark;
		facemark->loadModel(props.landmarksDetectionModel);
	}

	bool compute(frame_sp buffer)
	{
		faceDetector.detectMultiScale(iImg, faces);

		if (faces.size() == 0) {
			return false;
		}

	    bool success = facemark->fit(iImg, faces, landmarks);
		 
		return success;
	}

private:
	cv::CascadeClassifier faceDetector;
	cv::Ptr<cv::face::Facemark> facemark;
};
 
FacialLandmarkCV::FacialLandmarkCV(FacialLandmarkCVProps _props) : Module(TRANSFORM, "FacialLandmarkCV", _props), mProp( _props) {}

FacialLandmarkCV::~FacialLandmarkCV() {}

bool FacialLandmarkCV::validateInputPins()
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
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool FacialLandmarkCV::validateOutputPins()
{
	if (getNumberOfOutputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 2. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FACE_LANDMARKS_INFO && frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or FACE_LANDMARKS_INFO. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void FacialLandmarkCV::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	auto landmarksOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FACE_LANDMARKS_INFO));
	mOutputPinId1 = addOutputPin(landmarksOutputMetadata);
    rawFramePinId = addOutputPin(metadata);
}

bool FacialLandmarkCV::init()
{
	if (!Module::init())
    {
		return false;
	}
	if (mProp.type == FacialLandmarkCVProps::FaceDetectionModelType::SSD)
	{
		mDetail.reset(new DetailSSD(mProp));
	}

	else if (mProp.type == FacialLandmarkCVProps::FaceDetectionModelType::HAAR_CASCADE)
	{
		mDetail.reset(new DetailHCASCADE(mProp));
	}

	else
	{
		throw std::runtime_error("Invalid face detection model type");
	}

	return Module::init();
}

bool FacialLandmarkCV::term()
{
	mDetail.reset();
	return Module::term();
}

bool FacialLandmarkCV::process(frame_container& frames)
{
	auto frame = frames.cbegin()->second;
	bool computeValue = mDetail->compute(frame);

	if (computeValue == false) {
        send(frames);
        return true;
	}

	std::vector<RectangleOverlay> rectangleOverlays;

	for (const auto& face :mDetail->faces) {
		RectangleOverlay rectangleOverlay;
		rectangleOverlay.x1 = face.x;
		rectangleOverlay.y1 = face.y;
		rectangleOverlay.x2 = face.x + face.width;
		rectangleOverlay.y2 = face.y + face.height;

		rectangleOverlays.push_back(rectangleOverlay);
    }

	std::vector<CircleOverlay> circleOverlays;

	// Convert the landmarks from cv::Point2f to ApraPoint2f
	vector<vector<ApraPoint2f>> apralandmarks;
	for (const auto& landmark : mDetail->landmarks) {
		vector<ApraPoint2f> apralandmark;
		for (const auto& point : landmark) {
			apralandmark.emplace_back(ApraPoint2f(point));

			CircleOverlay circleOverlay;
			circleOverlay.x1 = point.x;
			circleOverlay.y1 = point.y;
			circleOverlay.radius = 1;

            circleOverlays.push_back(circleOverlay);
		}
		apralandmarks.emplace_back(std::move(apralandmark));
	}

	size_t bufferSize = sizeof(apralandmarks);
	for (auto i = 0; i < apralandmarks.size(); ++i)
	{
		bufferSize += sizeof(apralandmarks[i]) + (sizeof(ApraPoint2f) + 2 * sizeof(int)) * apralandmarks[i].size();
	}

	CompositeOverlay compositeOverlay;

	for (auto &rectangleOverlay : rectangleOverlays) {
		compositeOverlay.add(&rectangleOverlay);
    }

	for (auto &circleOverlay : circleOverlays) {
		compositeOverlay.add(&circleOverlay);
	}

	auto rawFrame = frames.cbegin()->second;    

	frames.insert(make_pair(rawFramePinId, rawFrame));

	if (rectangleOverlays.size() > 0 || circleOverlays.size() > 0) {
		DrawingOverlay drawingOverlay;
		drawingOverlay.add(&compositeOverlay);
		auto mvSize = drawingOverlay.mGetSerializeSize();
		auto landmarksFrame = makeFrame(mvSize, mOutputPinId1);
		drawingOverlay.serialize(landmarksFrame);
		frames.insert(make_pair(mOutputPinId1, landmarksFrame));
	}

	mDetail->faces.clear();
    mDetail->landmarks.clear();

	send(frames);

	return true;
}

bool FacialLandmarkCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->initMatImages(metadata);
	mDetail->setMetadata(metadata);
	mDetail->iImg.data = static_cast<uint8_t*>(frame->data());
	return true;
}

bool FacialLandmarkCV::shouldTriggerSOS()
{
	return mDetail->mFrameLength == 0;
}

bool FacialLandmarkCV::processEOS(string &pinId)
{
	mDetail->mFrameLength = 0;
	return true;
}

void FacialLandmarkCV::setProps(FacialLandmarkCVProps& props)
{
	Module::addPropsToQueue(props);
}

FacialLandmarkCVProps FacialLandmarkCV::getProps()
{
	fillProps(mDetail->props);
	return mDetail->props;
}

bool FacialLandmarkCV::handlePropsChange(frame_sp& frame)
{
	FacialLandmarkCVProps props(mDetail->props.type);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	if (props.type == FacialLandmarkCVProps::FaceDetectionModelType::SSD)
	{
		mDetail.reset(new DetailSSD(props));
	}
	else if (props.type == FacialLandmarkCVProps::FaceDetectionModelType::HAAR_CASCADE)
	{
		mDetail.reset(new DetailHCASCADE(props));
	}
	else
	{
		throw std::runtime_error("Invalid face detection model type");
	}
	return ret;
}