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
	bool init() override;
	bool term() override;
protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;
	std::string addOutputPin(framemetadata_sp& metadata);

private:
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	ImageResizeCVProps mProps;
	class Detail;
	std::shared_ptr<Detail> mDetail;
	size_t mMaxStreamLength;
};