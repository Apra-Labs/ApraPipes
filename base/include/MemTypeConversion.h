#pragma once

#include "Module.h"
#include "CudaCommon.h"

class DetailMemory;
class DetailDEVICEtoHOST;
class DetailHOSTtoDEVICE;
class DetailDMAtoHOST;
class DetailHOSTtoDMA;
class DetailDEVICEtoDMA;
class DetailDMAtoDEVICE;

class MemTypeConversionProps : public ModuleProps
{
public:
	MemTypeConversionProps(FrameMetadata::MemType _outputMemType) : ModuleProps()
	{
		outputMemType = _outputMemType;
	}

	MemTypeConversionProps(FrameMetadata::MemType _outputMemType, cudastream_sp &_stream) : ModuleProps()
	{
		outputMemType = _outputMemType;
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	cudastream_sp stream_sp;
	cudaStream_t stream;
	FrameMetadata::MemType outputMemType;
};

class MemTypeConversion : public Module
{
public:
	MemTypeConversion(MemTypeConversionProps _props);
	virtual ~MemTypeConversion();
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override; // throws exception if validation fails
	bool processEOS(std::string_view pinId) override;
	std::shared_ptr<DetailMemory> mDetail;
	MemTypeConversionProps mProps;
};
