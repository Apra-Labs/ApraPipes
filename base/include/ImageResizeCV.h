#pragma once
#include "FrameMetadata.h"
#include "Module.h"

class ImageResizeCVProps : public ModuleProps
{
public:
	ImageResizeCVProps(int _width, int _height)
	{
		width = _width;
		height = _height;
	}
	int width;
	int height;
};

class ImageResizeCV : public Module
{

public:
	ImageResizeCV(ImageResizeCVProps _props);
	virtual ~ImageResizeCV();
	bool init();
	bool term();
	void setProps(ImageResizeCVProps &props);
	ImageResizeCVProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	std::string addOutputPin(framemetadata_sp& metadata);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	ImageResizeCVProps mProps;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	size_t mMaxStreamLength;
};