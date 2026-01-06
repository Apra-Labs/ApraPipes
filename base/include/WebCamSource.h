#pragma once

#include "Module.h"
#include "declarative/PropertyMacros.h"

class WebCamSourceProps : public ModuleProps
{
public:
	// Constructor with defaults - supports declarative pipeline (can be called with no args)
	WebCamSourceProps(int _cameraId = -1, uint32_t _width = 640, uint32_t _height = 480, int _fps = 30) : ModuleProps(), width(_width), height(_height), cameraId(_cameraId), fps(_fps) {}

	uint32_t width = 640;
	uint32_t height = 480;
	int cameraId = -1;
	int fps = 30;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(width) + sizeof(height) + sizeof(cameraId) + sizeof(fps);
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
		apra::applyProp(props.cameraId, "cameraId", values, false, missingRequired);
		apra::applyProp(props.width, "width", values, false, missingRequired);
		apra::applyProp(props.height, "height", values, false, missingRequired);
		apra::applyProp(props.fps, "fps", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "cameraId") return static_cast<int64_t>(cameraId);
		if (propName == "width") return static_cast<int64_t>(width);
		if (propName == "height") return static_cast<int64_t>(height);
		if (propName == "fps") return static_cast<int64_t>(fps);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		// All properties are static (can't change after init)
		return false;
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};  // No dynamically changeable properties
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &width &height &cameraId &fps;
	}
};

class WebCamSource : public Module
{
public:
	WebCamSource(WebCamSourceProps props);
	virtual ~WebCamSource();
	bool init();
	bool term();

	void setProps(WebCamSourceProps &props);
	WebCamSourceProps getProps();

protected:
	bool produce();
	bool validateOutputPins();
	bool handlePropsChange(frame_sp &frame);

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};