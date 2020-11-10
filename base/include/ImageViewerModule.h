#pragma once

#include "Module.h"

class ImageViewerModuleProps : public ModuleProps
{
public:
	ImageViewerModuleProps(const string& _strTitle) : ModuleProps()
	{
		strTitle = _strTitle;
	}

	string strTitle;
};

class ImageViewerModule : public Module {
public:
	ImageViewerModule(ImageViewerModuleProps _props=ImageViewerModuleProps(""));
	virtual ~ImageViewerModule();
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




