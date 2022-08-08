#pragma once

#include "Module.h"

class NvTransformProps : public ModuleProps
{
public:
	NvTransformProps(ImageMetadata::ImageType _imageType, int _noOfframesToCapture) : top(0) , left(0) , width(0) , height(0)
	{
		imageType = _imageType;
		noOfframesToCapture = _noOfframesToCapture;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height, int _noOfframesToCapture) : top(0) , left(0) , width(_width) , height(_height)
	{
		imageType = _imageType;
		noOfframesToCapture = _noOfframesToCapture;
	}
	NvTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height,  int _top , int _left, int _noOfframesToCapture) : top(_top) , left(_left) , width(_width) , height(_height)
	{
		imageType = _imageType;
		noOfframesToCapture = _noOfframesToCapture;
	}
	ImageMetadata::ImageType imageType;	
	int top,left,width,height;
	int noOfframesToCapture;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(imageType) + 5 * sizeof(top);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &top;
		ar &left;
		ar &width;
		ar &height;
		ar &imageType;
		ar &noOfframesToCapture;
	}
};

class NvTransform : public Module
{

public:
	NvTransform(NvTransformProps _props);
	virtual ~NvTransform();
	bool init();
	bool term();
	bool resetFrameCapture();
	void setProps(NvTransformProps& props);
	NvTransformProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails
	bool processEOS(string& pinId);
	bool handleCommand(Command::CommandType type, frame_sp &frame);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	class NvTransformResetCommands;
};