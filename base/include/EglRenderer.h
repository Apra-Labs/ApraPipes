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
	EglRendererProps(uint32_t _x_offset,uint32_t _y_offset,bool _displayOnTop = true) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
		height = 0;
		width = 0;
		displayOnTop = _displayOnTop ? 1 : 0;
	}
    uint32_t x_offset;
    uint32_t y_offset;
	uint32_t height;
	uint32_t width;
	bool displayOnTop;
};

class EglRenderer : public Module
{
public:
    EglRenderer(EglRendererProps props);
    ~EglRenderer();

    bool init();
    bool term();
	bool closeWindow();
	bool createWindow(int width, int height);
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool handleCommand(Command::CommandType type, frame_sp& frame);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};