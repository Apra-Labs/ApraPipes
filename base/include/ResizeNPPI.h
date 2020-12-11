#pragma once

#include "Module.h"
#include "CudaCommon.h"

class ResizeNPPIProps : public ModuleProps
{
public:
	ResizeNPPIProps(int _width, int _height, cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		width = _width;
		height = _height;
		eInterpolation = 4; // NPPI_INTER_CUBIC
	}

	int width;
	int height;
	cudaStream_t stream;
	cudastream_sp stream_sp;
	int eInterpolation;
};

class ResizeNPPI : public Module
{

public:
	ResizeNPPI(ResizeNPPIProps _props);
	virtual ~ResizeNPPI();
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

	int mFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	ResizeNPPIProps props;		
};
