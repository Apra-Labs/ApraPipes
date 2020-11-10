#pragma once

#include "Module.h"
#include <cuda_runtime_api.h>
#include "FrameMetadata.h"

class CudaMemCopyProps : public ModuleProps
{
public:
	CudaMemCopyProps(cudaMemcpyKind kind, cudaStream_t _stream) : ModuleProps()
	{	
		alignLength = 0;
		memcpyKind = kind;
		stream = _stream;
		sync = false;
		if (memcpyKind == cudaMemcpyDeviceToHost)
		{
			sync = true;
		}
		else if (memcpyKind == cudaMemcpyHostToDevice)
		{
			alignLength = 512;
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Untested code. <" + std::to_string(memcpyKind) + ">");
		}
	}

	cudaMemcpyKind memcpyKind;
    cudaStream_t stream;
    bool sync;
	size_t alignLength;
};

class CudaMemCopy : public Module
{
public:
	CudaMemCopy(CudaMemCopyProps props);
	virtual ~CudaMemCopy();

	virtual bool init();
	virtual bool term();

	CudaMemCopyProps getProps();

protected:
	bool process(frame_container &frames);    
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
    void setOutputMetadata(framemetadata_sp& inputMetadata);
	framemetadata_sp cloneMetadata(framemetadata_sp metadata, FrameMetadata::MemType memType);

	CudaMemCopyProps props;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
	bool mCopy2D;
	FrameMetadata::MemType mMemType;
	FrameMetadata::FrameType mFrameType;
	size_t mSrcPitch[4];
	size_t mDstPitch[4];
	int mChannels;
	size_t mSrcNextPtrOffset[4];
	size_t mDstNextPtrOffset[4];
	size_t mRowSize[4];
	size_t mHeight[4];
};
