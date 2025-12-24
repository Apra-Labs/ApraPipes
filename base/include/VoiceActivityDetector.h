#pragma once

#include "Module.h"

class VoiceActivityDetectorProps : public ModuleProps
{
public:

	//decides speech vs no-speech
	enum AggressivenessMode {
		QUALITY = 0,          // Least aggressive (best quality, catches more speech)
		LOW_BITRATE = 1,      // Moderate
		AGGRESSIVE = 2,       // More aggressive
		VERY_AGGRESSIVE = 3   // Most aggressive (best bandwidth saving, only clear speech)
	};
	
	// audio time the VAD analyzes at once
	// default sample rate is 16 kHz (16,000 samples/sec)
	// 10ms - 160 samples  20ms - 320 samples  30ms - 480 samples
	enum FrameLength {
		FRAME_10MS = 10,      // Lowest latency
		FRAME_20MS = 20,      // Balanced
		FRAME_30MS = 30       // Best accuracy
	};
	
	VoiceActivityDetectorProps(
		int _sampleRate = 16000,
		AggressivenessMode _mode = QUALITY,
		FrameLength _frameLength = FRAME_10MS,
		bool _speechOnly = false  // When true, only outputs audio when speech is detected
	);
	
	int sampleRate;
	AggressivenessMode mode;
	FrameLength frameLength;
	bool speechOnly;  // If true, audio passthrough only sends frames with speech
	
	size_t getSerializeSize();

private:
	friend class boost::serialization::access;
	
	template <class Archive>
	void serialize(Archive& ar, const unsigned int version);
};

class VoiceActivityDetector : public Module
{
public:
	VoiceActivityDetector(VoiceActivityDetectorProps _props);
	virtual ~VoiceActivityDetector();
	
	bool init();
	bool term();
	void setProps(VoiceActivityDetectorProps& props);
	VoiceActivityDetectorProps getProps();

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
