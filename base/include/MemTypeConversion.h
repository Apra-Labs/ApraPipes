#pragma once

#include "Module.h"
#include "CudaCommon.h"
#include <memory>

class DetailMemory;
class DetailDEVICEtoHOST;
class DetailHOSTtoDEVICE;

#if defined(__arm__) || defined(__aarch64__)
class DetailDMAtoHOST;
class DetailHOSTtoDMA;
class DetailDEVICEtoDMA;
class DetailDMAtoDEVICE;
#endif

class MemTypeConversionProps : public ModuleProps
{
public:

	MemTypeConversionProps(FrameMetadata::MemType _outputMemType) : ModuleProps()
	{
        outputMemType = _outputMemType;
	}

	MemTypeConversionProps(FrameMetadata::MemType _outputMemType, cudastream_sp& _stream) : ModuleProps()
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
	bool init();
	bool term();
protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool processEOS(string &pinId);
	std::shared_ptr<DetailMemory> mDetail;
	MemTypeConversionProps mProps;
};
