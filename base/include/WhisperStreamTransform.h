#pragma once

#include "Module.h"

// size of audio to process should be a parameter. 
// Cache variable to collect frames for processing

class WhisperStreamTransformProps : public ModuleProps
{
public:
	enum DecoderSamplingStrategy {
		GREEDY,      //WHISPER_SAMPLING_GREEDY
		BEAM_SEARCH //WHISPER_SAMPLING_BEAM_SEARCH
	};
	WhisperStreamTransformProps(
		DecoderSamplingStrategy _samplingStrategy,
		std::string _modelPath,
		int _bufferSize) : samplingStrategy(_samplingStrategy),
		modelPath(_modelPath),
		bufferSize(_bufferSize)
	{}
    DecoderSamplingStrategy samplingStrategy;
    std::string modelPath;
	int bufferSize;
};

class WhisperStreamTransform  : public Module
{

public:
	WhisperStreamTransform(WhisperStreamTransformProps _props);
	virtual ~WhisperStreamTransform();
	bool init();
	bool term();
	void setProps(WhisperStreamTransformProps& props);
	WhisperStreamTransformProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};