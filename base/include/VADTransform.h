#pragma once

#include "Module.h"

class VADTransformProps : public ModuleProps
{
public:
	enum AggressivenessMode {
		QUALITY = 0,          // Least aggressive (best quality, catches more speech)
		LOW_BITRATE = 1,      // Moderate
		AGGRESSIVE = 2,       // More aggressive
		VERY_AGGRESSIVE = 3   // Most aggressive (best bandwidth saving, only clear speech)
	};
	
	enum FrameLength {
		FRAME_10MS = 10,
		FRAME_20MS = 20,
		FRAME_30MS = 30
	};
	
	VADTransformProps(
		int _sampleRate = 16000,
		AggressivenessMode _mode = QUALITY,
		FrameLength _frameLength = FRAME_10MS
	);
	
	int sampleRate;
	AggressivenessMode mode;
	FrameLength frameLength;
	
	size_t getSerializeSize();

private:
	friend class boost::serialization::access;
	
	template <class Archive>
	void serialize(Archive& ar, const unsigned int version);
};

class VADTransform : public Module
{
public:
	VADTransform(VADTransformProps _props);
	virtual ~VADTransform();
	
	bool init();
	bool term();
	void setProps(VADTransformProps& props);
	VADTransformProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
