#pragma once
#include <string>
#include "Module.h"

using namespace std;

class OverlayDetailAbs;
class DetailFFmpeg;
class DetailOpenh264;

class OverlayMotionVectorProps : public ModuleProps
{
public:
	enum MVOverlayMethod
	{
		FFMPEG,
		OPENH264
	};

	OverlayMotionVectorProps(MVOverlayMethod _MVOverlayMethod) : MVOverlay(_MVOverlayMethod)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
	MVOverlayMethod MVOverlay;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& MVOverlay;
	}
};

class  OverlayMotionVector : public Module
{
public:
	OverlayMotionVector(OverlayMotionVectorProps _props);
	virtual ~OverlayMotionVector() {};
	bool init();
	bool term();
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool processSOS(frame_sp& frame);
	bool shouldTriggerSOS();
private:
	boost::shared_ptr<OverlayDetailAbs> mDetail;
	std::string mOutputPinId;
};
