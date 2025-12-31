#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class TextOverlayXFormProps : public ModuleProps
{
public:
	TextOverlayXFormProps(double _alpha, std::string _text, std::string _position, bool _isDateTime, int _fontSize, std::string _fontColor, std::string _backgroundColor) : alpha(_alpha), text(_text), position(_position), isDateTime(_isDateTime), fontSize(_fontSize), fontColor(_fontColor), backgroundColor(_backgroundColor)
	{
	}

	TextOverlayXFormProps() : alpha(1.0), text(""), position("top-left"), isDateTime(false), fontSize(24), fontColor("white"), backgroundColor("transparent")
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

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.text, "text", values, false, missingRequired);
		apra::applyProp(props.position, "position", values, false, missingRequired);
		apra::applyProp(props.fontColor, "fontColor", values, false, missingRequired);
		apra::applyProp(props.backgroundColor, "backgroundColor", values, false, missingRequired);
		apra::applyProp(props.alpha, "alpha", values, false, missingRequired);
		apra::applyProp(props.isDateTime, "isDateTime", values, false, missingRequired);
		apra::applyProp(props.fontSize, "fontSize", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "text") return text;
		if (propName == "position") return position;
		if (propName == "fontColor") return fontColor;
		if (propName == "backgroundColor") return backgroundColor;
		if (propName == "alpha") return alpha;
		if (propName == "isDateTime") return isDateTime;
		if (propName == "fontSize") return static_cast<int64_t>(fontSize);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "text") { text = std::get<std::string>(value); return true; }
		if (propName == "position") { position = std::get<std::string>(value); return true; }
		if (propName == "fontColor") { fontColor = std::get<std::string>(value); return true; }
		if (propName == "backgroundColor") { backgroundColor = std::get<std::string>(value); return true; }
		if (propName == "alpha") { alpha = std::get<double>(value); return true; }
		if (propName == "isDateTime") { isDateTime = std::get<bool>(value); return true; }
		if (propName == "fontSize") { fontSize = static_cast<int>(std::get<int64_t>(value)); return true; }
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {"text", "position", "fontColor", "backgroundColor", "alpha", "isDateTime", "fontSize"};
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
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};