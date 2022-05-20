#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include <boost/pool/object_pool.hpp>

class DeviceToDMAMonoProps : public ModuleProps
{
public:
	DeviceToDMAMonoProps(cudastream_sp &_stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	cudastream_sp stream_sp;
	cudaStream_t stream;
};

class DeviceToDMAMono : public Module
{

public:
	DeviceToDMAMono(DeviceToDMAMonoProps _props);
	virtual ~DeviceToDMAMono();
	bool init();
	bool term();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	void setMetadata(framemetadata_sp &metadata);
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	DeviceToDMAMonoProps props;
	size_t mSrcPitch[4];
	size_t mDstPitch[4];
	int mChannels;
	size_t mSrcNextPtrOffset[4];
	size_t mDstNextPtrOffset[4];
	size_t mRowSize[4];
	size_t mHeight[4];
};