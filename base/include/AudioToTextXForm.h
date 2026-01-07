#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

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

	// Default constructor for declarative pipeline
	AudioToTextXFormProps() : samplingStrategy(GREEDY), modelPath(""), bufferSize(16000)
	{
	}

	AudioToTextXFormProps(
		DecoderSamplingStrategy _samplingStrategy,
		std::string _modelPath,
		int _bufferSize);
	size_t getSerializeSize();

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.modelPath, "modelPath", values, true, missingRequired);
		apra::applyProp(props.bufferSize, "bufferSize", values, false, missingRequired);

		// Handle enum via string
		auto it = values.find("samplingStrategy");
		if (it != values.end()) {
			if (auto* val = std::get_if<std::string>(&it->second)) {
				if (*val == "GREEDY") {
					props.samplingStrategy = GREEDY;
				} else if (*val == "BEAM_SEARCH") {
					props.samplingStrategy = BEAM_SEARCH;
				}
			}
		}
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "modelPath") return modelPath;
		if (propName == "bufferSize") return static_cast<int64_t>(bufferSize);
		if (propName == "samplingStrategy") {
			return samplingStrategy == GREEDY ? std::string("GREEDY") : std::string("BEAM_SEARCH");
		}
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		return false;  // All properties are static
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}

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
	bool init();
	bool term();
	void setProps(AudioToTextXFormProps& props);
	AudioToTextXFormProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);
	bool processEOS(string &pinId);
	bool handleFlushingBuffer();

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
