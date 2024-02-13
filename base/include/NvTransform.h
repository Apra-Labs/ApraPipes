#pragma once

#include "Module.h"

class CropRect : public ModuleProps
{
public:
	CropRect(int _top, int _left, int _width, int _height) : top(_top), left(_left), width(_width), height(_height) {}

	int top;
	int left;
	int width;
	int height;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 4;
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &top &left &width &height ;
	}	
};

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

	enum NvTransformRotate
	{
		ROTATE_0 = 0, // rotate 0 degress
		ANTICLOCKWISE_ROTATE_90, // rotate 90 degrees anticlockwise
		ANTICLOCKWISE_ROTATE_180, // rotate 180 degrees anticlockwise
		ANTICLOCKWISE_ROTATE_270, // rotate 270 degrees anticlockwise
		CLOCKWISE_ROTATE_90, // rotate 90 degrees clockwise
		CLOCKWISE_ROTATE_180, // rotate 180 degrees clockwise
		CLOCKWISE_ROTATE_270, // rotate 270 degrees clockwise
	};

	NvTransformProps(ImageMetadata::ImageType _imageType) : output_width(0), output_height(0), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = NvTransformRotate::ROTATE_0;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, NvTransformRotate _rotateType) : output_width(0), output_height(0), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height) : output_width(_width) , output_height(_height), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = NvTransformRotate::ROTATE_0;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, NvTransformRotate _rotateType) : output_width(_width), output_height(_height), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(_width) , output_height(_height), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = _filterType;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType) : output_width(_width) , output_height(_height), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
		rotateType = NvTransformRotate::ROTATE_0;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(_width) , output_height(_height), src_rect({0,0,0,0}), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, CropRect _src_rect) : output_width(_width) , output_height(_height), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = NvTransformRotate::ROTATE_0;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, CropRect _src_rect, NvTransformRotate _rotateType) : output_width(_width) , output_height(_height), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		scaleHeight = 1;
		scaleWidth = 1;
		filterType = NvTransformFilter::SMART;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, CropRect _src_rect, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType) : output_width(_width) , output_height(_height), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
		rotateType = NvTransformRotate::ROTATE_0;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, CropRect _src_rect, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(0) , output_height(0), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, CropRect _src_rect, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(0) , output_height(0), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = 1;
		scaleWidth = 1;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, CropRect _src_rect, float _scaleWidth, float _scaleHeight, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(_width) , output_height(_height), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = _scaleHeight;
		scaleWidth = _scaleWidth;
		rotateType = _rotateType;
	}

	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, CropRect _src_rect, NvTransformFilter _filterType, NvTransformRotate _rotateType) : output_width(_width) , output_height(_height), src_rect(_src_rect), dst_rect({0,0,0,0})
	{
		imageType = _imageType;
		filterType = _filterType;
		scaleHeight = 1;
		scaleWidth = 1;
		rotateType = _rotateType;
	}

	ImageMetadata::ImageType imageType;	
	int output_width, output_height;
	NvTransformFilter filterType;
	float scaleWidth, scaleHeight; // scaleWidth and scaleHeight are factor of width and height , 
								   //1 means no change 0.5 means half of actual dimension ,2 means twice of actual dimension

	NvTransformRotate rotateType;
	CropRect src_rect, dst_rect;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(float) * 2 + sizeof(src_rect) * 2 + sizeof(imageType) + sizeof(filterType) + sizeof(rotateType);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &output_width &output_height ;
		ar &scaleWidth &scaleHeight ;
		ar &src_rect &dst_rect ;
		ar& imageType;
		ar& filterType;
		ar& rotateType;
	}	
};

class NvTransform : public Module
{

public:
	NvTransform(NvTransformProps _props);
	virtual ~NvTransform();
	bool init();
	bool term();

	void setProps(NvTransformProps& props);
	NvTransformProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails
	bool processEOS(string& pinId);

	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};