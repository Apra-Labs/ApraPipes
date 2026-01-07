#pragma once
#include "Module.h"
#include <map>
#include <vector>
#include "declarative/PropertyMacros.h"

class ValveModuleProps : public ModuleProps
{
public:
	ValveModuleProps()
	{

	}

	ValveModuleProps(uint64 _noOfFramesToCapture)
	{
		noOfFramesToCapture = _noOfFramesToCapture;
	}
	uint64 noOfFramesToCapture = 10;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(noOfFramesToCapture);
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
		apra::applyProp(props.noOfFramesToCapture, "noOfFramesToCapture", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "noOfFramesToCapture") return static_cast<int64_t>(noOfFramesToCapture);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "noOfFramesToCapture") {
			return apra::applyFromVariant(noOfFramesToCapture, value);
		}
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {"noOfFramesToCapture"};
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& noOfFramesToCapture;
	}
};

class ValveModule : public Module
{
public:
	ValveModule(ValveModuleProps _props);
	~ValveModule();
	bool init();
	bool term();
	/* We can set the number of frames property by passing as
	arguement to allowFrames else module props value is taken */
	bool allowFrames(int numframes); 
	void setProps(ValveModuleProps& props);
	ValveModuleProps getProps();
	bool setNext(boost::shared_ptr<Module> next, bool open = true, bool sieve = false);
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};