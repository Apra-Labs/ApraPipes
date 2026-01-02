#pragma once

#include "Module.h"

// size of audio to process should be a parameter. 
// Cache variable to collect frames for processing

class AudioToTextXFormProps : public ModuleProps
{
public:
	enum DecoderSamplingStrategy {
		GREEDY,  
		BEAM_SEARCH
	};
	
	DecoderSamplingStrategy samplingStrategy;
	std::string modelPath;
	int bufferSize;

	AudioToTextXFormProps(
		DecoderSamplingStrategy _samplingStrategy,
		std::string _modelPath,
		int _bufferSize);
	size_t getSerializeSize();


private:
	friend class boost::serialization::access;
	
	template <class Archive>
	void serialize(Archive& ar, const unsigned int version);
};

class AudioToTextXForm  : public Module
{

public:
	AudioToTextXForm(AudioToTextXFormProps _props);
	virtual ~AudioToTextXForm();
	bool init() override;
	bool term() override;
	void setProps(AudioToTextXFormProps& props);
	AudioToTextXFormProps getProps();

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;
	bool handlePropsChange(frame_sp& frame) override;
	bool processEOS(std::string_view pinId);
	bool handleFlushingBuffer();

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
