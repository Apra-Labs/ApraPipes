#include "VADTransform.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "fvad.h"

// VADTransformProps implementation
VADTransformProps::VADTransformProps(
	int _sampleRate,
	AggressivenessMode _mode,
	FrameLength _frameLength
) : sampleRate(_sampleRate),
	mode(_mode),
	frameLength(_frameLength)
{}

size_t VADTransformProps::getSerializeSize() {
	return ModuleProps::getSerializeSize() +
		sizeof(sampleRate) +
		sizeof(mode) +
		sizeof(frameLength);
}

template <class Archive>
void VADTransformProps::serialize(Archive& ar, const unsigned int version) {
	ar& boost::serialization::base_object<ModuleProps>(*this);
	ar& sampleRate;
	ar& mode;
	ar& frameLength;
}

// Detail class - holds libfvad instance
class VADTransform::Detail
{
public:
	Detail(VADTransformProps& _props) : mProps(_props), mVad(nullptr)
	{
	}
	
	~Detail()
	{
		if (mVad) {
			fvad_free(mVad);
			mVad = nullptr;
		}
	}
	
	bool init()
	{
		// Create libfvad instance
		mVad = fvad_new();
		if (!mVad) {
			LOG_ERROR << "Failed to create libfvad instance";
			return false;
		}
		
		// Set sample rate (must be 8000, 16000, 32000, or 48000)
		if (fvad_set_sample_rate(mVad, mProps.sampleRate) < 0) {
			LOG_ERROR << "Invalid sample rate: " << mProps.sampleRate;
			LOG_ERROR << "Valid rates are: 8000, 16000, 32000, 48000";
			return false;
		}
		
		// Set aggressiveness mode (0-3)
		if (fvad_set_mode(mVad, mProps.mode) < 0) {
			LOG_ERROR << "Invalid aggressiveness mode: " << mProps.mode;
			return false;
		}
		
		LOG_INFO << "=== VAD Configuration ===";
		LOG_INFO << "Sample Rate: " << mProps.sampleRate << " Hz";
		LOG_INFO << "Aggressiveness Mode: " << mProps.mode << " (0=Quality, 3=Very Aggressive)";
		LOG_INFO << "Frame Length: " << mProps.frameLength << " ms";
		LOG_INFO << "Expected samples per frame: " << (mProps.sampleRate * mProps.frameLength / 1000);
		LOG_INFO << "========================";
		
		mFrameCount = 0;  // Reset frame counter
		
		return true;
	}
	
	int processAudio(const int16_t* samples, size_t length)
	{
		if (!mVad) {
			LOG_ERROR << "VAD not initialized";
			return -1;
		}
		// Returns: 1 = speech, 0 = silence, -1 = error
		int result = fvad_process(mVad, samples, length);
		
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
	
	void setProps(VADTransformProps& props)
	{
		mProps = props;
	}

public:
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	VADTransformProps mProps;

private:
	Fvad* mVad; 
	size_t mFrameCount;  
};

// VADTransform implementation
VADTransform::VADTransform(VADTransformProps _props) 
	: Module(TRANSFORM, "VADTransform", _props)
{
	mDetail.reset(new Detail(_props));
}

VADTransform::~VADTransform() {}

bool VADTransform::validateInputPins()
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

bool VADTransform::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1) {
		LOG_ERROR << "<" << getId() << ">::validateOutputPins expects 1 output. Actual: " << getNumberOfOutputPins();
		return false;
	}
	
	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();

	if (frameType != FrameMetadata::GENERAL) {
		LOG_ERROR << "<" << getId() << ">::validateOutputPins expects GENERAL output. Actual: " << frameType;
		return false;
	}
	
	return true;
}

void VADTransform::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	mDetail->mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool VADTransform::init()
{
	if (!Module::init()) {
		return false;
	}
	
	return mDetail->init();
}

bool VADTransform::term()
{
	return Module::term();
}

bool VADTransform::process(frame_container& frames)
{
	// 1. Get input audio frame
	auto frame = frames.begin()->second;
	int16_t* samples = static_cast<int16_t*>(frame->data());
	size_t sampleCount = frame->size() / 2;  // Int16 = 2 bytes
	
	// 2. Call libfvad
	int result = mDetail->processAudio(samples, sampleCount);
	
	// 3. Create output frame with VAD result
	auto outFrame = makeFrame(sizeof(int));
	int vadResult = (result == 1) ? 1 : 0;  
	memcpy(outFrame->data(), &vadResult, sizeof(int));
	
	// 4. Send output
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	
	return true;
}

bool VADTransform::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void VADTransform::setMetadata(framemetadata_sp& metadata)
{
	if (!metadata->isSet()) {
		return;
	}
}

VADTransformProps VADTransform::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

void VADTransform::setProps(VADTransformProps& props)
{
	Module::addPropsToQueue(props);
}

bool VADTransform::handlePropsChange(frame_sp& frame)
{
	auto ret = Module::handlePropsChange(frame, mDetail->mProps);
	mDetail->setProps(mDetail->mProps);
	
	mDetail->init();
	
	return ret;
}

bool VADTransform::processEOS(string& pinId)
{
	return true;
}
