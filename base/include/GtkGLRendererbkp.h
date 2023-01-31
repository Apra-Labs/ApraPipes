#pragma once

#include "Module.h"

class GtkGlRendererProps : public ModuleProps
{
public:
	GtkGlRendererProps(std::string _gladeFileName, int _windowWidth, int _windowHeight) : ModuleProps()
	{
		gladeFileName = _gladeFileName;
		windowWidth = _windowWidth;
		windowHeight = _windowHeight;
	}
	std::string gladeFileName;
	int windowWidth, windowHeight;
};

class GtkGlRenderer : public Module
{
public:
    GtkGlRenderer(GtkGlRendererProps props);
    ~GtkGlRenderer();

    bool init();
    bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	// bool shouldTriggerSOS();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};