#pragma once

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

	FacialLandmarkCVProps() {}

	FacialLandmarkCVProps(FaceDetectionModelType _type) : type(_type) {}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _modelConfiguration, const std::string _modelBinary, const std::string _landmarksDetectionModel)
		: type(_type), modelConfiguration(_modelConfiguration), modelBinary(_modelBinary), landmarksDetectionModel(_landmarksDetectionModel)
	{
	}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _faceDetectionModel,const std::string _landmarksDetectionModel)
		: type(_type), landmarksDetectionModel(_landmarksDetectionModel), faceDetectionModel(_faceDetectionModel)
	{
	}

	FaceDetectionModelType type;
	const std::string modelConfiguration = "./data/deploy.prototxt.txt";
	const std::string modelBinary = "./data/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	const std::string landmarksDetectionModel = "./data/face_landmark_model.dat";
	const std::string faceDetectionModel = "./data/haarcascade.xml";

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
	bool intializer(FacialLandmarkCVProps props);
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