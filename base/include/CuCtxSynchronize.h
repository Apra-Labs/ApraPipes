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

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();	
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:
	CuCtxSynchronizeProps props;
}; 