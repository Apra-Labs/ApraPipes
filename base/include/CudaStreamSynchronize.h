#pragma once

#include "Module.h"
#include <cuda_runtime_api.h>

class CudaStreamSynchronizeProps : public ModuleProps
{
public:
	CudaStreamSynchronizeProps(cudaStream_t _stream) : ModuleProps() 
	{
		stream = _stream;
	}
	
	cudaStream_t stream;
};

class CudaStreamSynchronize : public Module {
public:

	CudaStreamSynchronize(CudaStreamSynchronizeProps _props);
	virtual ~CudaStreamSynchronize() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();	
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:
	CudaStreamSynchronizeProps props;
};



