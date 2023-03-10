#pragma once
#include "Module.h"

using namespace std;
class DetailAbs;
class DetailFfmpeg;
class DetailOpenH264;

class MotionVectorExtractorProps : public ModuleProps
{
public:
	enum MVExtractMethod
	{
		FFMPEG,
		OPENH264
	};

	MotionVectorExtractorProps(MVExtractMethod _MVExtractMethod = MVExtractMethod::FFMPEG , bool _sendDecodedFrame = false) : MVExtract(_MVExtractMethod), sendDecodedFrame(_sendDecodedFrame)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(sendDecodedFrame);
	}
	bool sendDecodedFrame = false;
	MVExtractMethod MVExtract = MVExtractMethod::FFMPEG;
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
	boost::shared_ptr<DetailAbs> mDetail;
	std::string motionVectorPinId;
	framemetadata_sp rawOutputMetadata;
	bool mShouldTriggerSOS = true;
};
