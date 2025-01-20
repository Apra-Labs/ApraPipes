#pragma once
#include "Module.h"

using namespace std;
class MvExtractDetailAbs;
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

	MotionVectorExtractorProps(MVExtractMethod _MVExtractMethod = MVExtractMethod::FFMPEG, int _streamWidth = -1, int _streamHeight = -1, bool _sendDecodedFrame = false, int _motionVectorThreshold = 1, int _minMotionVectorRequired = 1) : MVExtract(_MVExtractMethod), sendDecodedFrame(_sendDecodedFrame), motionVectorThreshold(_motionVectorThreshold), minMotionVectorRequired(_minMotionVectorRequired), streamWidth(_streamWidth), streamHeight(_streamHeight)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(sendDecodedFrame) + sizeof(motionVectorThreshold) + sizeof(MVExtract) + sizeof(minMotionVectorRequired) + sizeof(streamWidth)+ sizeof(streamHeight);
	}
	bool sendDecodedFrame = false;
	int motionVectorThreshold;
	MVExtractMethod MVExtract = MVExtractMethod::FFMPEG;
	int minMotionVectorRequired;
	int streamWidth;
	int streamHeight;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& sendDecodedFrame;
		ar& motionVectorThreshold;
		ar& MVExtract;
		ar& minMotionVectorRequired;
		ar& streamWidth;
		ar& streamHeight;
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
	MotionVectorExtractorProps getProps();
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp& frame);
	void setMetadata(frame_sp metadata);
	bool handlePropsChange(frame_sp& frame);
private:
	boost::shared_ptr<MvExtractDetailAbs> mDetail;
	framemetadata_sp rawOutputMetadata;
	bool mShouldTriggerSOS = true;
};
