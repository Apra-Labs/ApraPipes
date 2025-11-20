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

	int mFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	ResizeNPPIProps props;		
};
