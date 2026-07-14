#pragma once

#include <atomic>

#include "Module.h"
#include <vector>

class VideoDecoderProps : public ModuleProps
{
public:
	VideoDecoderProps(uint _lowerWaterMark = 300, uint _upperWaterMark = 350)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
	}
	uint lowerWaterMark;
	uint upperWaterMark;
};

class VideoDecoder : public Module
{
public:
	VideoDecoder(VideoDecoderProps _props);
	virtual ~VideoDecoder();
	bool init();
	bool term();
	bool processEOS(string& pinId);
	bool isEOSProcessed() const { return mEOSProcessed.load(); }
	int getConsecutiveErrors() const;

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	void flushQue();
	bool handleCommand(Command::CommandType type, frame_sp& frame);

private:
	void sendDecodedFrames();

	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::atomic<bool> mEOSProcessed{false};
	bool mShouldTriggerSOS;
	bool mHelperReady;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	VideoDecoderProps mProps;

};
