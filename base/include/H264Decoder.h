#pragma once

#include "Module.h"
#include <cuda_runtime_api.h>
#include "CudaCommon.h"

class H264DecoderNvCodecProps : public ModuleProps
{
public:
	H264DecoderNvCodecProps() {}
};

class H264DecoderNvCodec : public Module
{
public:
	H264DecoderNvCodec(H264DecoderNvCodecProps _props);
	virtual ~H264DecoderNvCodec();
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
	H264DecoderNvCodecProps props;
};
