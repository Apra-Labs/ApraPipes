#pragma once

#include <opencv2/face.hpp>
#include "Module.h"

class Detail;
class DetailSSD;
class DetailHCASCADE;

class FacialLandmarkCVProps : public ModuleProps
{
public:
	enum FaceDetectionModelType
	{
		SSD,
		HAAR_CASCADE
	};

	FacialLandmarkCVProps(FaceDetectionModelType _type) : type(_type) {}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _modelConfiguration, const std::string _modelBinary, const std::string _landmarksDetectionModel, cv::Ptr<cv::face::Facemark> _facemark)
		: type(_type), modelConfiguration(_modelConfiguration), modelBinary(_modelBinary), landmarksDetectionModel(_landmarksDetectionModel),facemark(_facemark)
	{
		if (_type != FaceDetectionModelType::SSD)
		{
			throw AIPException(AIP_FATAL, "This constructor only supports SSD");
		}
	}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _faceDetectionModel,const std::string _landmarksDetectionModel, cv::Ptr<cv::face::Facemark> _facemark)
		: type(_type), landmarksDetectionModel(_landmarksDetectionModel), faceDetectionModel(_faceDetectionModel), facemark(_facemark)
	{
		if (_type != FaceDetectionModelType::HAAR_CASCADE)
		{
			throw AIPException(AIP_FATAL, "This constructor only supports HAAR_CASCADE ");
		}
	}

	FaceDetectionModelType type;
	const std::string modelConfiguration = "./data/assets/deploy.prototxt";
	const std::string modelBinary = "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	const std::string landmarksDetectionModel = "./data/assets/face_landmark_model.dat";
	const std::string faceDetectionModel = "./data/assets/haarcascade.xml";
	cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkKazemi::create();

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(type);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& type;
	}
};

class FacialLandmarkCV : public Module
{
 public:
	FacialLandmarkCV(FacialLandmarkCVProps props);
	virtual ~FacialLandmarkCV();
	bool init();
	bool term();
	void setProps(FacialLandmarkCVProps& props);
	FacialLandmarkCVProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	boost::shared_ptr<Detail> mDetail;
	FacialLandmarkCVProps mProp;
	bool handlePropsChange(frame_sp& frame);
	std::string mOutputPinId1;
};