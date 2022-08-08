#pragma once

#include "Module.h"

class TextOverlayDMAProps : public ModuleProps
{
public:

    TextOverlayDMAProps(int _xCoordinate, int _yCoordinate) : text(""), xCoordinate(_xCoordinate), yCoordinate(_yCoordinate)
	{
	}
	TextOverlayDMAProps(std::string _text, int _xCoordinate, int _yCoordinate) : text(_text), xCoordinate(_xCoordinate), yCoordinate(_yCoordinate)
	{
	}
	std::string text;
	int xCoordinate, yCoordinate;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(string);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &xCoordinate &yCoordinate &text;
	}
};

class TextOverlayDMA : public Module
{

public:
	TextOverlayDMA(TextOverlayDMAProps _props);
	virtual ~TextOverlayDMA();
	bool init();
	bool term();
	void setProps(TextOverlayDMAProps &props);
	TextOverlayDMAProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId);
	void setProps(TextOverlayDMA);
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};