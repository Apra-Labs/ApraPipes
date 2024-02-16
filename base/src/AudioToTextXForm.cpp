#include "AudioToTextXForm.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "whisper.h"
#include "SFML/Config.hpp"

AudioToTextXFormProps::AudioToTextXFormProps(
	DecoderSamplingStrategy _samplingStrategy,
	std::string _modelPath,
	int _bufferSize) : samplingStrategy(_samplingStrategy),
	modelPath(_modelPath),
	bufferSize(_bufferSize)
{}

size_t AudioToTextXFormProps::getSerializeSize() {
	return ModuleProps::getSerializeSize() +
		sizeof(samplingStrategy) +
		sizeof(modelPath) +
		sizeof(bufferSize);
}

template <class Archive>
void AudioToTextXFormProps::serialize(Archive& ar, const unsigned int version) {
	ar& boost::serialization::base_object<ModuleProps>(*this);
	ar& samplingStrategy;
	ar& modelPath;
	ar& bufferSize;
}

class AudioToTextXForm::Detail
{
public:
	Detail(AudioToTextXFormProps& _props) : mProps(_props)
	{
	}
	~Detail() {}

	whisper_full_params fetchDefaultParams() {
		auto samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
		switch (mProps.samplingStrategy)
		{
		case AudioToTextXFormProps::DecoderSamplingStrategy::GREEDY:
			samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
			break;
		case AudioToTextXFormProps::DecoderSamplingStrategy::BEAM_SEARCH:
			samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH;
			break;
		default:
			throw AIPException(AIP_FATAL, "Unknown Sampling Strategy");
		}
		return whisper_full_default_params(samplingStrategy);
	}

	void setProps(AudioToTextXFormProps& props)
	{
		mProps = props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	std::vector<float> mInputAudioBuffer;
	AudioToTextXFormProps mProps;
	int mFrameType;
	whisper_context *mWhisperContext = NULL;
	whisper_full_params mWhisperFullParams;
	whisper_context_params mWhisperContextParams;
	int toleranceBufferSize = 16000; //16000 is 1 second worth of samples since data is captured at 16KHz
};

AudioToTextXForm::AudioToTextXForm(AudioToTextXFormProps _props) : Module(TRANSFORM, "AudioToTextXForm", _props)
{
	mDetail.reset(new Detail(_props));
}

AudioToTextXForm::~AudioToTextXForm() {}

bool AudioToTextXForm::validateInputPins()
{
	//TODO: Reject any audio pin that has a samplingRate!=16KHz 
	//https://github.com/Apra-Labs/ApraPipes/issues/325
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();

	FrameMetadata::FrameType frameType = metadata->getFrameType();

	if (frameType != FrameMetadata::AUDIO)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Audio. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}
	return true;
}

bool AudioToTextXForm::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::TEXT)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be TEXT. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

void AudioToTextXForm::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::TEXT));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool AudioToTextXForm::init()
{
	mDetail->mInputAudioBuffer.reserve(mDetail->mProps.bufferSize + mDetail->toleranceBufferSize);
	mDetail->mWhisperFullParams = mDetail->fetchDefaultParams();
	mDetail->mWhisperContextParams = whisper_context_default_params();
	mDetail->mWhisperContext = whisper_init_from_file_with_params(mDetail->mProps.modelPath.c_str(), mDetail->mWhisperContextParams);
	return Module::init();
}

bool AudioToTextXForm::term()
{
	whisper_free_context_params(&mDetail->mWhisperContextParams);
	whisper_free_params(&mDetail->mWhisperFullParams);
	whisper_free(mDetail->mWhisperContext);
	return Module::term();
}

bool AudioToTextXForm::process(frame_container& frames)
{
	auto frame = frames.begin()->second;
	sf::Int16* constFloatPointer = static_cast<sf::Int16*>(frame->data());
	int numberOfSamples = frame->size() / 2;
	//TODO: Modify to use NPP/ IPP
	for (int index = 0; index < numberOfSamples; index++) {
		mDetail->mInputAudioBuffer.push_back((float)constFloatPointer[index]/ 32768.0f);
	}
	if (mDetail->mInputAudioBuffer.size() < mDetail->mProps.bufferSize) {
		return true;
	}
	whisper_full(
		mDetail->mWhisperContext,
		mDetail->mWhisperFullParams,
		mDetail->mInputAudioBuffer.data(),
		mDetail->mInputAudioBuffer.size()
	);
	std::string output = "";
	const int n_segments = whisper_full_n_segments(mDetail->mWhisperContext);
	for (int i = 0; i < n_segments; ++i) {
		const char* text = whisper_full_get_segment_text(mDetail->mWhisperContext, i);
		output += text;
	}
	mDetail->mInputAudioBuffer.clear();
	auto outFrame = makeFrame(output.length());
	memcpy(outFrame->data(), output.c_str(), output.length());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	return true;
}

void AudioToTextXForm::setMetadata(framemetadata_sp& metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
}

bool AudioToTextXForm::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

AudioToTextXFormProps AudioToTextXForm::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

bool AudioToTextXForm::handlePropsChange(frame_sp& frame)
{
	AudioToTextXFormProps props(mDetail->mProps.samplingStrategy, mDetail->mProps.modelPath, mDetail->mProps.bufferSize);
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	mDetail->mWhisperFullParams = mDetail->fetchDefaultParams();
	return ret;
}

void AudioToTextXForm::setProps(AudioToTextXFormProps& props)
{
	if (props.modelPath != mDetail->mProps.modelPath) {
		throw AIPException(AIP_FATAL, "Model Path dynamic change not handled");
	}
	Module::addPropsToQueue(props);
}