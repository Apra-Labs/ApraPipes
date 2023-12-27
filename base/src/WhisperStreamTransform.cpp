#include "WhisperStreamTransform.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "whisper.h"
#include "SFML/Config.hpp"

class WhisperStreamTransform::Detail
{
public:
	Detail(WhisperStreamTransformProps& _props) : mProps(_props)
	{
	}
	~Detail() {}

	void setProps(WhisperStreamTransformProps& props)
	{
		mProps = props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	std::vector<float> inputAudioBuffer;
	WhisperStreamTransformProps mProps;
	int mFrameType;
	whisper_context *mWhisperContext = NULL;
	whisper_full_params mWhisperFullParams;
	whisper_context_params mWhisperContextParams;
};

WhisperStreamTransform::WhisperStreamTransform(WhisperStreamTransformProps _props) : Module(TRANSFORM, "WhisperStreamTransform", _props)
{
	mDetail.reset(new Detail(_props));
}

WhisperStreamTransform::~WhisperStreamTransform() {}

bool WhisperStreamTransform::validateInputPins()
{
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

	return true;
}

bool WhisperStreamTransform::validateOutputPins()
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

void WhisperStreamTransform::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::TEXT));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool WhisperStreamTransform::init()
{
	//intialize model
	auto samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
	switch (mDetail->mProps.samplingStrategy)
	{
		case WhisperStreamTransformProps::DecoderSamplingStrategy::GREEDY:
			samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
			break;
		case WhisperStreamTransformProps::DecoderSamplingStrategy::BEAM_SEARCH:
			samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH;
			break;
		default:
			samplingStrategy = whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
	}
	mDetail->mWhisperFullParams = whisper_full_default_params(samplingStrategy);
	mDetail->mWhisperContextParams = whisper_context_default_params();
	mDetail->mWhisperContext = whisper_init_from_file_with_params(mDetail->mProps.modelPath.c_str(), mDetail->mWhisperContextParams);
	return Module::init();
}

bool WhisperStreamTransform::term()
{
	whisper_free_context_params(&mDetail->mWhisperContextParams);
	whisper_free_params(&mDetail->mWhisperFullParams);
	whisper_free(mDetail->mWhisperContext);
	return Module::term();
}

bool WhisperStreamTransform::process(frame_container& frames)
{
	auto frame = frames.begin()->second;
	sf::Int16* constFloatPointer = static_cast<sf::Int16*>(frame->data());
	int numberOfSamples = frame->size() / 2;
	for (int index = 0; index < numberOfSamples; index++) {
		mDetail->inputAudioBuffer.push_back((float)constFloatPointer[index]/ 32768.0f);
	}
	if (mDetail->inputAudioBuffer.size() < mDetail->mProps.bufferSize) {
		return true;
	}
	whisper_full(
		mDetail->mWhisperContext,
		mDetail->mWhisperFullParams,
		mDetail->inputAudioBuffer.data(),
		mDetail->inputAudioBuffer.size()
	);
	std::string output = "";
	const int n_segments = whisper_full_n_segments(mDetail->mWhisperContext);
	for (int i = 0; i < n_segments; ++i) {
		const char* text = whisper_full_get_segment_text(mDetail->mWhisperContext, i);
		output += text;
	}
	mDetail->inputAudioBuffer.clear();
	auto outFrame = makeFrame(output.length());
	memcpy(outFrame->data(), output.c_str(), output.length());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	return true;
}

void WhisperStreamTransform::setMetadata(framemetadata_sp& metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
}

bool WhisperStreamTransform::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

WhisperStreamTransformProps WhisperStreamTransform::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

bool WhisperStreamTransform::handlePropsChange(frame_sp& frame)
{
	WhisperStreamTransformProps props(mDetail->mProps.samplingStrategy, mDetail->mProps.modelPath,32000);
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void WhisperStreamTransform::setProps(WhisperStreamTransformProps& props)
{
	Module::addPropsToQueue(props);
}