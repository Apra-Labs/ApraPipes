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
		LOG_INFO << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool CudaStreamSynchronize::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_INFO << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
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
	// auto startTime = std::chrono::high_resolution_clock::now(); // Start timer

	auto cudaStatus = cudaStreamSynchronize(props.stream);
	if (cudaStatus != cudaSuccess)
	{
		LOG_INFO << "cudaStreamSynchronize failed <" << cudaStatus << ">";
	}

	send(frames);

	// auto endTime = std::chrono::high_resolution_clock::now(); // End timer
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(); // Calculate duration in milliseconds

    // LOG_DEBUG << "Time taken to process frame: " << duration << " ms";

	return true;
}