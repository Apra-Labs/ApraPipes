#include "VoiceActivityDetector.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "fvad.h"

// VoiceActivityDetectorProps implementation
VoiceActivityDetectorProps::VoiceActivityDetectorProps(
	int _sampleRate,
	AggressivenessMode _mode,
	FrameLength _frameLength,
	bool _speechOnly
) : sampleRate(_sampleRate),
	mode(_mode),
	frameLength(_frameLength),
	speechOnly(_speechOnly)
{}

size_t VoiceActivityDetectorProps::getSerializeSize() {
	return ModuleProps::getSerializeSize() +
		sizeof(sampleRate) +
		sizeof(mode) +
		sizeof(frameLength) +
		sizeof(speechOnly);
}

template <class Archive>
void VoiceActivityDetectorProps::serialize(Archive& ar, const unsigned int version) {
	ar& boost::serialization::base_object<ModuleProps>(*this);
	ar& sampleRate;
	ar& mode;
	ar& frameLength;
	ar& speechOnly;
}

class VoiceActivityDetector::Detail
{
public:
	Detail(VoiceActivityDetectorProps& _props) : mProps(_props), mVoiceDetector(nullptr)
	{
	}
	
	~Detail()
	{
		if (mVoiceDetector) {
			fvad_free(mVoiceDetector);
			mVoiceDetector = nullptr;
		}
	}
	
	bool init()
	{
		// Create libfvad instance
		mVoiceDetector = fvad_new();
		if (!mVoiceDetector) {
			LOG_ERROR << "Failed to create libfvad instance";
			return false;
		}
		
		// Set sample rate (must be 8000, 16000, 32000, or 48000)
		if (fvad_set_sample_rate(mVoiceDetector, mProps.sampleRate) < 0) {
			LOG_ERROR << "Invalid sample rate: " << mProps.sampleRate;
			LOG_ERROR << "Valid rates are: 8000, 16000, 32000, 48000";
			return false;
		}
		
		// Set aggressiveness mode (0-3)
		if (fvad_set_mode(mVoiceDetector, mProps.mode) < 0) {
			LOG_ERROR << "Invalid aggressiveness mode: " << mProps.mode;
			return false;
		}
		
		LOG_INFO << "=== VAD Configuration ===";
		LOG_INFO << "Sample Rate: " << mProps.sampleRate << " Hz";
		LOG_INFO << "Aggressiveness Mode: " << mProps.mode << " (0=Quality, 3=Very Aggressive)";
		LOG_INFO << "Frame Length: " << mProps.frameLength << " ms";
		LOG_INFO << "Speech Only Mode: " << (mProps.speechOnly ? "ON" : "OFF");
		LOG_INFO << "Expected samples per frame: " << (mProps.sampleRate * mProps.frameLength / 1000);
		LOG_INFO << "========================";
		
		mFrameCount = 0;  // Reset frame counter
		return true;
	}
	
	int processAudio(const int16_t* samples, size_t length)
	{
		if (!mVoiceDetector) {
			LOG_ERROR << "VAD not initialized";
			return -1;
		}
		// Returns: 1 = speech, 0 = silence, -1 = error
		int result = fvad_process(mVoiceDetector, samples, length);
		
		if (result < 0) {
			LOG_ERROR << "fvad_process failed. Check frame length matches expected size.";
			LOG_ERROR << "Expected samples for " << mProps.frameLength << "ms at " 
			          << mProps.sampleRate << "Hz: " 
			          << (mProps.sampleRate * mProps.frameLength / 1000);
			LOG_ERROR << "Actual samples received: " << length;
		}
		
		mFrameCount++;
		
		return result;
	}
	
	void setProps(VoiceActivityDetectorProps& props)
	{
		mProps = props;
	}

public:
	// Detail class definition
	framemetadata_sp mAudioMetadata;
	framemetadata_sp mVadMetadata;
	std::string mAudioPinId;
	std::string mVadPinId;
	VoiceActivityDetectorProps mProps;

private:
	Fvad* mVoiceDetector; 
	size_t mFrameCount;  
};

VoiceActivityDetector::VoiceActivityDetector(VoiceActivityDetectorProps _props) 
	: Module(TRANSFORM, "VoiceActivityDetector", _props)
{
	mDetail.reset(new Detail(_props));
}

VoiceActivityDetector::~VoiceActivityDetector() {}

bool VoiceActivityDetector::validateInputPins()
{
	if (getNumberOfInputPins() != 1) {
		LOG_ERROR << "<" << getId() << ">::validateInputPins expects 1 input. Actual: " << getNumberOfInputPins();
		return false;
	}
	
	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	
	if (frameType != FrameMetadata::AUDIO) {
		LOG_ERROR << "<" << getId() << ">::validateInputPins expects AUDIO input. Actual: " << frameType;
		return false;
	}
	
	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST) {
		LOG_ERROR << "<" << getId() << ">::validateInputPins expects HOST memory. Actual: " << memType;
		return false;
	}
	
	return true;
}

bool VoiceActivityDetector::validateOutputPins()
{
	if (getNumberOfOutputPins() > 2) {
		LOG_ERROR << "<" << getId() << ">::validateOutputPins expects <= 2 outputs. Actual: " << getNumberOfOutputPins();
		return false;
	}
	
	return true;
}

void VoiceActivityDetector::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	
	// Pin 1: Audio Passthrough
	mDetail->mAudioMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::AUDIO, FrameMetadata::MemType::HOST));
	mDetail->mAudioMetadata->copyHint(*metadata.get());
	mDetail->mAudioPinId = addOutputPin(mDetail->mAudioMetadata);

	// Pin 2: VAD Result
	mDetail->mVadMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL, FrameMetadata::MemType::HOST));
	mDetail->mVadPinId = addOutputPin(mDetail->mVadMetadata);
}

bool VoiceActivityDetector::init()
{
	if (!Module::init()) {
		return false;
	}
	
	return mDetail->init();
}

bool VoiceActivityDetector::term()
{
	return Module::term();
}

bool VoiceActivityDetector::process(frame_container& frames)
{
	// 1. Get input audio frame
	auto frame = frames.begin()->second;
	int16_t* samples = static_cast<int16_t*>(frame->data());
	size_t sampleCount = frame->size() / 2;  // Int16 = 2 bytes
	
	// 2. Call libfvad
	int result = mDetail->processAudio(samples, sampleCount);
	bool isSpeech = (result == 1);
	
	// 3. Audio Passthrough - Conditional on speechOnly mode
	// speechOnly - 
	// false:audio has booth speech and silence
	// true:audio has only speech
	if (!mDetail->mProps.speechOnly ) {
		frames.insert(make_pair(mDetail->mAudioPinId, frame));
	}
	else {
		if(isSpeech) {
			frames.insert(make_pair(mDetail->mAudioPinId, frame));
		}
	}

	// 4. Create output frame with VAD result - ALWAYS create and send
	auto outFrame = makeFrame(sizeof(int), mDetail->mVadPinId);
	int vadResult = isSpeech ? 1 : 0;  
	memcpy(outFrame->data(), &vadResult, sizeof(int));
	
	// 5. Send output
	frames.insert(make_pair(mDetail->mVadPinId, outFrame));
	send(frames);
	
	return true;
}

bool VoiceActivityDetector::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void VoiceActivityDetector::setMetadata(framemetadata_sp& metadata)
{
	if (!metadata->isSet()) {
		return;
	}
}

VoiceActivityDetectorProps VoiceActivityDetector::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

void VoiceActivityDetector::setProps(VoiceActivityDetectorProps& props)
{
	Module::addPropsToQueue(props);
}

bool VoiceActivityDetector::handlePropsChange(frame_sp& frame)
{
	auto ret = Module::handlePropsChange(frame, mDetail->mProps);
	mDetail->setProps(mDetail->mProps);
	
	mDetail->init();
	
	return ret;
}

bool VoiceActivityDetector::processEOS(string& pinId)
{
	return true;
}
