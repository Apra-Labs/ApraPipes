#pragma once

#include "Module.h"
#include "CudaCommon.h"

class TensorRTProps: public ModuleProps
{
public:
	TensorRTProps(string _enginePath, cudastream_sp& _stream) : ModuleProps()
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
        enginePath = _enginePath;
	}
	cudastream_sp stream_sp;
	cudaStream_t stream;
	string enginePath;	
};

class TensorRT : public Module {
public:

	TensorRT(TensorRTProps props);	
	~TensorRT() {}

	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails
	void setMetadata(framemetadata_sp& metadata);
private:
    framemetadata_sp mOutputMetadata;
    string mOutputPinId;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};