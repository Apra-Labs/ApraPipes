#pragma once

#include "Module.h"
#include <cuda_runtime_api.h>

class JPEGEncoderNVJPEGProps : public ModuleProps
{
public:
	JPEGEncoderNVJPEGProps(cudaStream_t _stream)
	{
		stream = _stream;
		quality = 90;
	}

	unsigned short quality;
	cudaStream_t stream;
};

class JPEGEncoderNVJPEG : public Module
{

public:
	JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps _props);
	virtual ~JPEGEncoderNVJPEG();
	bool init();
	bool term();

	void getImageSize(int& width, int& height);

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

	size_t mMaxStreamLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};
