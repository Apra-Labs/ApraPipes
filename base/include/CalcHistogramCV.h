#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>
#include "declarative/PropertyMacros.h"

/*
ROI - optional
Mask - optional
numBins - number of bins
https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist

depth has to be either CV_8U or CV_32F
*/

class CalcHistogramCVProps: public ModuleProps
{
public:
	CalcHistogramCVProps() : ModuleProps()
	{
		bins = 8;
	}

	CalcHistogramCVProps(int _bins) : ModuleProps()
	{
		bins = _bins;
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(bins) + sizeof(roi) + sizeof(maskImgPath) + maskImgPath.length();
	}

	// All the properties can be updated during run time using setProps
	int bins;
	vector<int> roi;
	string maskImgPath;

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		apra::applyProp(props.bins, "bins", values, false, missingRequired);
		apra::applyProp(props.maskImgPath, "maskImgPath", values, false, missingRequired);
		// Note: roi (vector<int>) is not supported via scalar properties - use programmatic API
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "bins") return static_cast<int64_t>(bins);
		if (propName == "maskImgPath") return maskImgPath;
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		if (propName == "bins") { bins = static_cast<int>(std::get<int64_t>(value)); return true; }
		if (propName == "maskImgPath") { maskImgPath = std::get<std::string>(value); return true; }
		throw std::runtime_error("Unknown property: " + propName);
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {"bins", "maskImgPath"};
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);

		ar & bins;
		ar & roi;
		ar & maskImgPath;
	}
};

class CalcHistogramCV : public Module {
public:

	CalcHistogramCV(CalcHistogramCVProps props);	
	virtual ~CalcHistogramCV() {}

	virtual bool init();
	virtual bool term();

	void setProps(CalcHistogramCVProps& props);
	CalcHistogramCVProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();	
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails	
	bool shouldTriggerSOS();

	bool handlePropsChange(frame_sp& frame);

private:	
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};



