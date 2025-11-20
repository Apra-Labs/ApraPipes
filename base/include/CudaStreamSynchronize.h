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

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, string& pinId) override;

private:
	CudaStreamSynchronizeProps props;
};



