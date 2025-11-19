#pragma once

#include "Module.h"
#include "CudaCommon.h"

class RotateNPPIProps : public ModuleProps
{
public:
	RotateNPPIProps(cudastream_sp &_stream, double _angle)
	{
		stream = _stream;
		angle = _angle;
	}

	double angle;
	cudastream_sp stream;
};

class RotateNPPI : public Module
{

public:
	RotateNPPI(RotateNPPIProps props);
	virtual ~RotateNPPI();
	bool init();
	bool term();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
