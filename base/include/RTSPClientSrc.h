#pragma once
#include <chrono>
#include <string>
#include "Module.h"
#include "declarative/PropertyMacros.h"

class RTSPClientSrcProps : public ModuleProps
{
public:
	RTSPClientSrcProps(const std::string& rtspURL, const std::string& userName, const std::string& password, bool useTCP=true) : ModuleProps(),
		rtspURL(rtspURL), userName(userName), password(password),useTCP(useTCP)
	{
	}

	RTSPClientSrcProps() : ModuleProps(), useTCP(true)
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(rtspURL) + sizeof(userName) + sizeof(password), sizeof(useTCP);
	}

	string rtspURL, userName, password;
	bool useTCP = true;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.rtspURL, "rtspURL", values, true, missingRequired);
		apra::applyProp(props.userName, "userName", values, false, missingRequired);
		apra::applyProp(props.password, "password", values, false, missingRequired);
		apra::applyProp(props.useTCP, "useTCP", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "rtspURL") return rtspURL;
		if (propName == "userName") return userName;
		if (propName == "password") return password;
		if (propName == "useTCP") return useTCP;
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

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& rtspURL;
		ar& userName;
		ar& password;
		ar& useTCP;
	}
};

class RTSPClientSrc : public Module {
public:
	RTSPClientSrc(RTSPClientSrcProps _props);
	virtual ~RTSPClientSrc();
	bool init();
	bool term();
	void setProps(RTSPClientSrcProps& props);
	int getCurrentFps();
	RTSPClientSrcProps getProps();

protected:
	bool produce();
	bool validateOutputPins();
	void notifyPlay(bool play);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
private:
	RTSPClientSrcProps mProps;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
