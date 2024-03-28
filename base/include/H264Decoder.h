#pragma once

#include "Module.h"

class H264DecoderProps : public ModuleProps
{
public:
	H264DecoderProps() : ModuleProps()
	{}
};

class H264Decoder : public Module
{
public:
	H264Decoder(H264DecoderProps _props);
	virtual ~H264Decoder();
	bool init();
	bool term();
	bool processEOS(string& pinId);
    bool decoderEos();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool handleCommand(Command::CommandType type, frame_sp& frame);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	bool mShouldTriggerSOS;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	H264DecoderProps mProps;
};