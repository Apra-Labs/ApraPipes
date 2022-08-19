#pragma once

#include "Module.h"
#include "CudaCommon.h"

class H264EncoderNVCodecProps : public ModuleProps
{
public:
	enum H264CodecProfile
	{
		BASELINE,
		MAIN,
		HIGH,
	};

	H264EncoderNVCodecProps(uint32_t &_bitRateKbps, apracucontext_sp& _cuContext, uint32_t &_gopLength,uint32_t &_frameRate,H264CodecProfile _vProfile,uint32_t& _enableBFrames) : cuContext(_cuContext), gopLength(_gopLength)
	{
		gopLength = _gopLength;
		frameRate = _frameRate;
		bitRateKbps = _bitRateKbps;
		vProfile = _vProfile;
		enableBFrames = _enableBFrames;
	}
	H264EncoderNVCodecProps(apracucontext_sp& _cuContext) : bitRateKbps(0), cuContext(_cuContext)
	{

	}
	H264CodecProfile vProfile= H264EncoderNVCodecProps::BASELINE;
	uint32_t enableBFrames;
	uint32_t gopLength = 30;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	apracucontext_sp cuContext;
};

class H264EncoderNVCodec : public Module
{

public:
	H264EncoderNVCodec(H264EncoderNVCodecProps _props);
	virtual ~H264EncoderNVCodec();
	bool init();
	bool term();
	bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height);

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;

	bool mShouldTriggerSOS;
	framemetadata_sp mOutputMetadata;
	std::string mInputPinId;
	std::string mOutputPinId;

	H264EncoderNVCodecProps props;
};
