#pragma once

#include "Module.h"
#include "CudaCommon.h"

class JPEGEncoderNVJPEGProps : public ModuleProps
{
public:
	JPEGEncoderNVJPEGProps(cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		quality = 90;
	}

	unsigned short quality;
	cudaStream_t stream;
	cudastream_sp stream_sp;
};

class JPEGEncoderNVJPEG : public Module
{

public:
	JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps _props);
	virtual ~JPEGEncoderNVJPEG();
	bool init() override;
	bool term() override;

	void getImageSize(int& width, int& height);

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool shouldTriggerSOS() override;
	bool processEOS(string& pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;

	size_t mMaxStreamLength;
	std::string mOutputPinId;
};
