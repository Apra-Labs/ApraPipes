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
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override; // throws exception if validation fails
	bool shouldTriggerSOS() override;
	bool processEOS(std::string_view pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
