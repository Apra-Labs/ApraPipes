#pragma once

#include "Module.h"

class DetailRenderer;
class DetailEgl;
class DetailImageviewer;

class ImageViewerModuleProps : public ModuleProps
{
public:
#if 1
	ImageViewerModuleProps(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height, bool _displayOnTop = true) : ModuleProps()
	{
		x_offset = _x_offset;
		y_offset = _y_offset;
		height = _height;
		width = _width;
		displayOnTop = _displayOnTop ? 1 : 0;
	}
	ImageViewerModuleProps(uint32_t _x_offset, uint32_t _y_offset, bool _displayOnTop = true) : ModuleProps()
	{
		x_offset = _x_offset;
		y_offset = _y_offset;
		height = 0;
		width = 0;
		displayOnTop = _displayOnTop ? 1 : 0;
	}
	ImageViewerModuleProps(const string &_strTitle) : ModuleProps()
	{
		strTitle = _strTitle;
	}
#else
	ImageViewerModuleProps(const string &_strTitle) : ModuleProps()
	{
		strTitle = _strTitle;
	}
#endif

	uint32_t x_offset = 0;
	uint32_t y_offset = 0;
	uint32_t height;
	uint32_t width;
	bool displayOnTop;
	string strTitle;
};

class ImageViewerModule : public Module
{
public:
	ImageViewerModule(ImageViewerModuleProps _props);
	virtual ~ImageViewerModule();
	bool init();
	bool term();
	bool closeWindow();
	bool createWindow(int width, int height);

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	void addInputPin(framemetadata_sp &metadata, string &pinId);
	bool handleCommand(Command::CommandType type, frame_sp &frame);
	boost::shared_ptr<DetailRenderer> mDetail;
	ImageViewerModuleProps mProps;
};