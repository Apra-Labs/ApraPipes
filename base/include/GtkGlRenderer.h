#pragma once

#include "Module.h"
#include <chrono>
#include <gtk/gtk.h>

class GtkGlRendererProps : public ModuleProps
{
public:
	GtkGlRendererProps(GtkWidget *_glArea, int _windowWidth, int _windowHeight, bool _isPlaybackRenderer = true) : ModuleProps() // take gtk string
	{
		// gladeFileName = _gladeFileName;
		glArea = _glArea;
		windowWidth = _windowWidth;
		windowHeight = _windowHeight;
		isPlaybackRenderer = _isPlaybackRenderer;
	}
	GtkWidget *glArea;
	int windowWidth = 0;
	int windowHeight = 0;
	bool isPlaybackRenderer = true;
};

class GtkGlRenderer : public Module
{
public:
	GtkGlRenderer(GtkGlRendererProps props);
	~GtkGlRenderer();
	bool init();
	bool term();
	bool changeProps(GtkWidget *glArea, int windowWidth, int windowHeight);
protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool handleCommand(Command::CommandType type, frame_sp &frame);
	void pushFrame(frame_sp frame);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::chrono::steady_clock::time_point lastFrameTime =
		std::chrono::steady_clock::now();
	std::queue<frame_sp> frameQueue;
	std::mutex queueMutex;
};
