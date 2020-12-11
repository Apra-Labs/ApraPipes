#pragma once

#include "Module.h"
#include "CudaCommon.h"

class CudaStreamSynchronizeProps : public ModuleProps
{
public:
	CudaStreamSynchronizeProps(cudastream_sp& _stream) : ModuleProps() 
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}
	
	cudaStream_t stream;
	cudastream_sp stream_sp;
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



