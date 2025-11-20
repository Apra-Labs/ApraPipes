#pragma once

#include "Module.h"

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

	bool init() override;
	bool term() override;

protected:	
	bool process(frame_container& frames) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;

private:
    class Detail;
	std::shared_ptr<Detail> mDetail;
	KeyboardListenerProps props;
};