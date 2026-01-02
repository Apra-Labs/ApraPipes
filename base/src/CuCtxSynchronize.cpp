#include <boost/foreach.hpp>

#include "CuCtxSynchronize.h"
#include "cuda_runtime.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaDriverLoader.h"


CuCtxSynchronize::CuCtxSynchronize(CuCtxSynchronizeProps _props) :Module(TRANSFORM, "CuCtxSynchronize", _props), props(_props)
{
}

bool CuCtxSynchronize::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool CuCtxSynchronize::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

bool CuCtxSynchronize::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CuCtxSynchronize::term()
{
	return Module::term();
}

void CuCtxSynchronize::addInputPin(framemetadata_sp& metadata, std::string_view pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool CuCtxSynchronize::process(frame_container& frames)
{
	cudaFree(0);

	auto& loader = CudaDriverLoader::getInstance();
	if (!loader.isAvailable() || !loader.cuCtxSynchronize) {
		LOG_ERROR << "CUDA driver not available for cuCtxSynchronize";
		send(frames);
		return true; // Continue processing even if CUDA not available
	}

	auto cudaStatus = loader.cuCtxSynchronize();
	if (cudaStatus != CUDA_SUCCESS)
	{
		LOG_ERROR << "cuCtxSynchronize failed <" << cudaStatus << ">";
	}

	send(frames);

	return true;
}