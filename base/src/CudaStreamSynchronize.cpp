#include <boost/foreach.hpp>

#include "CudaStreamSynchronize.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"


CudaStreamSynchronize::CudaStreamSynchronize(CudaStreamSynchronizeProps _props) :Module(TRANSFORM, "CudaStreamSynchronize", _props), props(_props)
{
}

bool CudaStreamSynchronize::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool CudaStreamSynchronize::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

bool CudaStreamSynchronize::init()
{
	if (!Module::init())
	{
		return false;
	}


	return true;
}

bool CudaStreamSynchronize::term()
{
	return Module::term();
}

void CudaStreamSynchronize::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool CudaStreamSynchronize::process(frame_container& frames)
{
	auto cudaStatus = cudaStreamSynchronize(props.stream);
	if (cudaStatus != cudaSuccess)
	{
		LOG_ERROR << "cudaStreamSynchronize failed <" << cudaStatus << ">";
	}

	send(frames);

	return true;
}