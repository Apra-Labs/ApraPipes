#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>

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



