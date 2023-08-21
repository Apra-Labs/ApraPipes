#pragma once

#include "Module.h"

class H264DecoderProps : public ModuleProps
{
public:
	H264DecoderProps() {}
};

class H264Decoder : public Module
{
public:
	H264Decoder(H264DecoderProps _props);
	virtual ~H264Decoder();
	bool init();
	bool term();
	bool processEOS(string& pinId);

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	bool mShouldTriggerSOS;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	H264DecoderProps mProps;
	void bufferDecodedFrames(frame_sp& frame);
	void bufferEncodedFrames(frame_sp& frame);
	std::deque<std::pair<std::deque<frame_sp>, bool>> gop;
	std::deque<frame_sp> tempGop ;
	std::deque<std::deque<frame_sp>> bufferedDecodedFrames;
	std::deque<frame_sp> tempDecodedFrames;
	std::queue<std::pair<uint, bool>> framesInGopAndDirectionTracker;
	void sendDecodedFrame();
	uint framesCounterOfCurrentGop = 0;
	bool hasDirectionChangedToForward = false;
	bool hasDirectionChangedToBackward = false;
	bool foundGopIFrame = false;
	void sendFramesToDecoder();
};