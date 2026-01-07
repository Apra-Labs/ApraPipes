#pragma once
#include "Module.h"
#include "AIPExceptions.h"
#include "declarative/PropertyMacros.h"

class ArchiveSpaceManagerProps : public ModuleProps
{
public:
	// Default constructor for declarative pipeline - validation happens in init()
	ArchiveSpaceManagerProps() : lowerWaterMark(0), upperWaterMark(0), pathToWatch(""), samplingFreq(60)
	{
		fps = 0.001;
	}

	ArchiveSpaceManagerProps(uint64_t _lowerWaterMark, uint64_t _upperWaterMark, string _pathToWatch, int _samplingFreq)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
		pathToWatch = _pathToWatch;
		samplingFreq = _samplingFreq;
		fps = 0.001;

		validateProps();
	}

	ArchiveSpaceManagerProps(uint64_t maxSizeAllowed, string _pathToWatch, int _samplingFreq)
	{
		lowerWaterMark = maxSizeAllowed - (maxSizeAllowed / 10);
		upperWaterMark = maxSizeAllowed;
		pathToWatch = _pathToWatch;
		samplingFreq = _samplingFreq;
		fps = 0.001;

		validateProps();
	}

	// Validation method - can be called from constructor or init()
	void validateProps() const
	{
		if (pathToWatch.empty()) {
			throw AIPException(AIP_FATAL, "pathToWatch cannot be empty");
		}
		auto totalSpace = boost::filesystem::space(pathToWatch);
		if ((lowerWaterMark > upperWaterMark) || (upperWaterMark > totalSpace.capacity))
		{
			LOG_ERROR << "Please enter correct properties!";
			std::string errorMsg = "Incorrect properties set for Archive Manager. TotalDiskCapacity <" + std::to_string(totalSpace.capacity) + ">lowerWaterMark<" + std::to_string(lowerWaterMark) + "> UpperWaterMark<" + std::to_string(upperWaterMark) + ">";
			throw AIPException(AIP_FATAL, errorMsg);
		}
	}


	uint64_t lowerWaterMark; // Lower disk space
	uint64_t upperWaterMark; // Higher disk space
	std::string pathToWatch;
	int samplingFreq;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(lowerWaterMark) + sizeof(upperWaterMark) + sizeof(pathToWatch) + sizeof(samplingFreq);
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
		apra::applyProp(props.pathToWatch, "pathToWatch", values, true, missingRequired);
		// Handle uint64_t via int64_t conversion
		auto itLower = values.find("lowerWaterMark");
		if (itLower != values.end()) {
			if (auto* val = std::get_if<int64_t>(&itLower->second)) {
				props.lowerWaterMark = static_cast<uint64_t>(*val);
			}
		} else {
			missingRequired.push_back("lowerWaterMark");
		}
		auto itUpper = values.find("upperWaterMark");
		if (itUpper != values.end()) {
			if (auto* val = std::get_if<int64_t>(&itUpper->second)) {
				props.upperWaterMark = static_cast<uint64_t>(*val);
			}
		} else {
			missingRequired.push_back("upperWaterMark");
		}
		apra::applyProp(props.samplingFreq, "samplingFreq", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "pathToWatch") return pathToWatch;
		if (propName == "lowerWaterMark") return static_cast<int64_t>(lowerWaterMark);
		if (propName == "upperWaterMark") return static_cast<int64_t>(upperWaterMark);
		if (propName == "samplingFreq") return static_cast<int64_t>(samplingFreq);
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		// All properties are static (can't change watched path after init)
		return false;
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};  // No dynamically changeable properties
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& lowerWaterMark;
		ar& upperWaterMark;
		ar& pathToWatch;
		ar& samplingFreq;
	}
};


class ArchiveSpaceManager : public Module {
public:
	ArchiveSpaceManager(ArchiveSpaceManagerProps _props);

	virtual ~ArchiveSpaceManager() {
	}
	bool init();
	bool term();
	uint64_t finalArchiveSpace = 0;
	void setProps(ArchiveSpaceManagerProps& props);
	ArchiveSpaceManagerProps getProps();

protected:
	bool produce();
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);
private:

	class Detail;
	boost::shared_ptr<Detail> mDetail;
	bool checkDirectory = true;
};