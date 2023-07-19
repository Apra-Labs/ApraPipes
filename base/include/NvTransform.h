#pragma once

#include "Module.h"

class NvTransformProps : public ModuleProps
{
public:
	enum NvTransformFilter
	{
		NEAREST=0, // transform filter nearest.
		BILINEAR,  // transform filter bilinear.
		TAP_5,     // transform filter 5 tap.
		TAP_10,    // transform filter 10 tap.
		SMART,     // transform filter smart.
		NICEST     // transform filter nicest.
	};

	NvTransformProps(ImageMetadata::ImageType _imageType) : top(0) , left(0) , width(0) , height(0)
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height) : top(0) , left(0) , width(_width) , height(_height)
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType) : top(0) , left(0) , width(_width) , height(_height)
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height,  int _top , int _left) : top(_top) , left(_left) , width(_width) , height(_height)
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height,  int _top , int _left, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType) : top(_top) , left(_left) , width(_width) , height(_height)
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
	}
	ImageMetadata::ImageType imageType;	
	int top,left,width,height;
	NvTransformFilter filterType;
	float scaleWidth, scaleHeight; // scaleWidth and scaleHeight are factor of width and height , 
								   //1 means no change 0.5 means half of actual dimension ,2 means twice of actual dimension	
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