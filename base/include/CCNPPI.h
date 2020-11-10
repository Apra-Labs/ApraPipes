#pragma once

#include "Module.h"
#include <cuda_runtime_api.h>

// dynamic change of image type is not supported for now

class CCNPPIProps : public ModuleProps
{
public:
	CCNPPIProps(ImageMetadata::ImageType _imageType, cudaStream_t _stream)
	{
		stream = _stream;
		imageType = _imageType;
	}
	
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
