#pragma once

#include "Module.h"

class TextOverlayXFormProps : public ModuleProps
{
public:
	TextOverlayXFormProps(int _frameWidth, int _frameHeight, double _alpha, std::string _text, std::string _position, bool _isDateTime, int _fontSize, std::string _fontColor, std::string _backgroundColor) : frameWidth(_frameWidth), frameHeight(_frameHeight), alpha(_alpha), text(_text), position(_position), isDateTime(_isDateTime), fontSize(_fontSize), fontColor(_fontColor), backgroundColor(_backgroundColor)
	{
	}
	std::string text, fontColor, position, backgroundColor;
	double alpha;
	bool isDateTime;
	int fontSize, frameWidth, frameHeight;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 3 + sizeof(string) * 4 + sizeof(bool) + sizeof(double) + text.length() + fontColor.length() + position.length() + backgroundColor.length();
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &text &fontSize &fontColor &isDateTime &position &frameHeight &frameWidth &alpha &backgroundColor;
	}
};

class TextOverlayXForm : public Module
{

public:
	TextOverlayXForm(TextOverlayXFormProps _props);
	virtual ~TextOverlayXForm();
	bool init();
	bool term();
	void setProps(TextOverlayXFormProps &props);
	TextOverlayXFormProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId);
	void setProps(TextOverlayXForm);
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};