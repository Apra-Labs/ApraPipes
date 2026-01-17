#pragma once
#include "FrameMetadata.h"
#include "Module.h"
#include "declarative/PropertyMacros.h"

class ImageResizeCVProps : public ModuleProps
{
public:
	ImageResizeCVProps(int _width, int _height)
	{
		width = _width;
		height = _height;
	}

	ImageResizeCVProps() : width(0), height(0) {}

	int width;
	int height;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.width, "width", values, true, missingRequired);
		apra::applyProp(props.height, "height", values, true, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "width") return static_cast<int64_t>(width);
		if (propName == "height") return static_cast<int64_t>(height);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		throw std::runtime_error("Cannot modify static property '" + propName + "' after initialization");
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};
	}
};

class ImageResizeCV : public Module
{

public:
	ImageResizeCV(ImageResizeCVProps _props);
	virtual ~ImageResizeCV();
	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	std::string addOutputPin(framemetadata_sp& metadata);

private:
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	ImageResizeCVProps mProps;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	size_t mMaxStreamLength;
};