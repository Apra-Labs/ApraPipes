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
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override; // throws exception if validation fails
	bool shouldTriggerSOS() override;
	bool processEOS(std::string_view pinId) override;

private:
	void setMetadata(framemetadata_sp& metadata);

	class Detail;
	std::shared_ptr<Detail> mDetail;

	bool mNoChange;
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	framemetadata_sp mIntermediateMetadata;
	std::string mOutputPinId;
	CCNPPIProps mProps;		
};
