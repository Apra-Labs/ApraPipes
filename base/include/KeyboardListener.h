#pragma once

#include "Module.h"
#include "CudaCommon.h"

class KeyboardListenerProps : public ModuleProps
{
public:
	KeyboardListenerProps(uint8_t _NosFrame) : ModuleProps() 
	{
        nosFrame = _NosFrame;
	}
    uint8_t nosFrame;
};

class KeyboardListener : public Module {
public:

	KeyboardListener(KeyboardListenerProps _props);
	virtual ~KeyboardListener() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();	
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:
    class Detail;
	boost::shared_ptr<Detail> mDetail;
	KeyboardListenerProps props;
};