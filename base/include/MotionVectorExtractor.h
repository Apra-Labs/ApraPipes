#pragma once
#include "Module.h"

using namespace std;

class MotionVectorExtractorProps : public ModuleProps
{
public:
	MotionVectorExtractorProps(bool _sendDecodedFrame = false)
	{
		sendDecodedFrame = _sendDecodedFrame;
	}
	bool sendDecodedFrame = false;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(sendDecodedFrame);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& sendDecodedFrame;
	}
};

class MotionVectorExtractor : public Module
{
public:
	MotionVectorExtractor(MotionVectorExtractorProps _props);
	virtual ~MotionVectorExtractor() {};
	bool init();
	bool term();
	void setProps(MotionVectorExtractorProps& props);
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp& frame);
	void setMetadata(frame_sp metadata);
	bool handlePropsChange(frame_sp& frame);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::string motionVectorPinId;
	framemetadata_sp rawOutputMetadata;
	bool mShouldTriggerSOS = true;
};
