#pragma once
#include <string>
#include "Module.h"

using namespace std;

class MotionExtractorProps : public ModuleProps
{
public:
	MotionExtractorProps(bool _sendRgbFrame = false)
	{
		sendRgbFrame = _sendRgbFrame;
	}
	bool sendRgbFrame = false;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(sendRgbFrame);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& sendRgbFrame;
	}
};

class MotionExtractor : public Module
{
public:
	MotionExtractor(MotionExtractorProps _props);
	virtual ~MotionExtractor() {};
	bool init();
	bool term();
	void setProps(MotionExtractorProps& props);
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp& frame);
	void setMetadata(framemetadata_sp& metadata);
	bool handlePropsChange(frame_sp& frame);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::string motionVectorPinId;
	framemetadata_sp rawOutputMetadata;
};
