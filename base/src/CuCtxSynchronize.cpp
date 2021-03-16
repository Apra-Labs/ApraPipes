#include <boost/foreach.hpp>

#include "CuCtxSynchronize.h"
#include "cuda_runtime.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"


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

void CuCtxSynchronize::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool CuCtxSynchronize::process(frame_container& frames)
{
	auto cudaStatus = cuCtxSynchronize();
	if (cudaStatus != CUDA_SUCCESS)
	{
		LOG_ERROR << "cuCtxSynchronize failed <" << cudaStatus << ">";
	}

	send(frames);

	return true;
}

bool CuCtxSynchronize::processSOS(frame_sp &frame)
{
    cudaFree(0);
	return true;
}