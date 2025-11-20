#pragma once

#include "Module.h"

class TextOverlayXFormProps : public ModuleProps
{
public:
	TextOverlayXFormProps(double _alpha, std::string _text, std::string _position, bool _isDateTime, int _fontSize, std::string _fontColor, std::string _backgroundColor) : alpha(_alpha), text(_text), position(_position), isDateTime(_isDateTime), fontSize(_fontSize), fontColor(_fontColor), backgroundColor(_backgroundColor)
	{
	}
	std::string text, fontColor, position, backgroundColor;
	double alpha;
	bool isDateTime;
	int fontSize;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(fontSize) + sizeof(text) + sizeof(fontColor) + sizeof(position) + sizeof(backgroundColor) + sizeof(isDateTime) + sizeof(alpha) + text.length() + fontColor.length() + position.length() + backgroundColor.length();
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &text &fontSize &fontColor &isDateTime &position &alpha &backgroundColor;
	}
};

class TextOverlayXForm : public Module
{

public:
	TextOverlayXForm(TextOverlayXFormProps _props);
	virtual ~TextOverlayXForm();
	bool init() override;
	bool term() override;
	void setProps(TextOverlayXFormProps &props);
	TextOverlayXFormProps getProps();

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override;
	bool handlePropsChange(frame_sp &frame) override;

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	std::shared_ptr<Detail> mDetail;
};