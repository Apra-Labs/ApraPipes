#pragma once

#include "Module.h"
#include "CudaCommon.h"

class CuCtxSynchronizeProps : public ModuleProps
{
public:
	CuCtxSynchronizeProps() : ModuleProps() 
	{
	}
};

class CuCtxSynchronize : public Module {
public:

	CuCtxSynchronize(CuCtxSynchronizeProps _props);
	virtual ~CuCtxSynchronize() {}

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, string& pinId) override;

private:
	CuCtxSynchronizeProps props;
}; 