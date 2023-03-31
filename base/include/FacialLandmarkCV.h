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

	FacialLandmarkCVProps() : ModuleProps() {}

	FacialLandmarkCVProps(FaceDetectionModelType _type) : ModuleProps()
	{
		type = _type;
	}

	FaceDetectionModelType type;
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
	std::string mOutputPinId;
};