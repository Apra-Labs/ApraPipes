#pragma once

#include "Module.h"

class NvTransformProps : public ModuleProps
{
public:
	NvTransformProps(ImageMetadata::ImageType _imageType) : top(0) , left(0) , width(0) , height(0)
	{
		imageType = _imageType;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height) : top(0) , left(0) , width(_width) , height(_height)
	{
		imageType = _imageType;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height,  int _top , int _left) : top(_top) , left(_left) , width(_width) , height(_height)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
	int top,left,width,height;
};

class NvTransform : public Module
{

public:
	NvTransform(NvTransformProps _props);
	virtual ~NvTransform();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};