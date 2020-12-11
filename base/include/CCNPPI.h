#pragma once

#include "Module.h"
#include "CudaCommon.h"

// dynamic change of image type is not supported for now

class CCNPPIProps : public ModuleProps
{
public:
	CCNPPIProps(ImageMetadata::ImageType _imageType, cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		imageType = _imageType;
	}
	
	cudastream_sp stream_sp;
	cudaStream_t stream;
	ImageMetadata::ImageType imageType;	
};

class CCNPPI : public Module
{

public:
	CCNPPI(CCNPPIProps _props);
	virtual ~CCNPPI();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);

	class Detail;
	boost::shared_ptr<Detail> mDetail;

	bool mNoChange;
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	CCNPPIProps props;		
};
