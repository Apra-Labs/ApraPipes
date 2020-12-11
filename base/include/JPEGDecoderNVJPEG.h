#pragma once

#include "Module.h"
#include "ImageMetadata.h"
#include "CudaCommon.h"

class JPEGDecoderNVJPEGProps : public ModuleProps
{
public:
	JPEGDecoderNVJPEGProps(cudastream_sp& _stream)
	{
		imageType = ImageMetadata::UNSET;
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	cudaStream_t stream;
	cudastream_sp stream_sp;
	ImageMetadata::ImageType imageType; 
};

class JPEGDecoderNVJPEG : public Module
{

public:
	JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps _props);
	virtual ~JPEGDecoderNVJPEG();
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

	size_t mOutputSize;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};
