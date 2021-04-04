#pragma once

#include "Module.h"

class EglRendererProps : public ModuleProps
{
public:
	EglRendererProps(uint32_t _x_offset,uint32_t _y_offset, uint32_t _width, uint32_t _height) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
		height = _height;
		width = _width;
	}
	EglRendererProps(uint32_t _x_offset,uint32_t _y_offset) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
		height = 0;
		width = 0;
	}
    uint32_t x_offset;
    uint32_t y_offset;
	uint32_t height;
	uint32_t width;
};

class EglRenderer : public Module
{
public:
    EglRenderer(EglRendererProps props);
    ~EglRenderer();

    bool init();
    bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};