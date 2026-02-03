#pragma once

#include "Module.h"

class NvTransformProps : public ModuleProps
{
public:
    enum class NvRotation
    {
        None_ = 0,
        Rotate90 = 90,
        Rotate180 = 180,
        Rotate270 = 270
    };

    enum class NvFlip
    {
        None_ = 0,
        FlipX = 1,
        FlipY = 2
    };

    // Default crop constructor
    NvTransformProps(ImageMetadata::ImageType _imageType)
        : top(0), left(0), width(0), height(0),
          rotation(NvRotation::None_), flip(NvFlip::None_)
    {
        imageType = _imageType;
    }

    // Crop with width and height
    NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height)
        : top(0), left(0), width(_width), height(_height),
          rotation(NvRotation::None_), flip(NvFlip::None_)
    {
        imageType = _imageType;
    }

    // Crop with width, height, top, left
    NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, int _top, int _left)
        : top(_top), left(_left), width(_width), height(_height),
          rotation(NvRotation::None_), flip(NvFlip::None_)
    {
        imageType = _imageType;
    }

	// Rotation constructor
    NvTransformProps(ImageMetadata::ImageType _imageType, NvRotation _rotation)
        : top(0), left(0), width(0), height(0),
          rotation(_rotation),flip(NvFlip::None_)
    {
        imageType = _imageType;
    }

    // Flip constructor
    NvTransformProps(ImageMetadata::ImageType _imageType, NvFlip _flip)
        : top(0), left(0), width(0), height(0),
          rotation(NvRotation::None_), flip(_flip)
    {
        imageType = _imageType;
    }

	//crop with rotation
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, int _top, int _left, NvRotation _rotation)
	: top(_top), left(_left), width(_width), height(_height),
	 rotation(_rotation), flip(NvFlip::None_)
    {
        imageType = _imageType;
    }

	//crop with flip
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, int _top, int _left, NvFlip _flip)
	: top(_top), left(_left), width(_width), height(_height),
	rotation(NvRotation::None_), flip(_flip)
    {
        imageType = _imageType;
    }
   
	

    ImageMetadata::ImageType imageType;
    int top, left, width, height;
    NvRotation rotation;
    NvFlip flip;
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