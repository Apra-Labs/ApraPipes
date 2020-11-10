#pragma once

#include "Module.h"

/*
ROI - optional
Mask - optional
numBins - number of bins
https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist

depth has to be either CV_8U or CV_32F
*/

class HistogramOverlayProps : public ModuleProps
{
public:
	HistogramOverlayProps() : ModuleProps() {}
};

class HistogramOverlay : public Module {
public:

	HistogramOverlay(HistogramOverlayProps _props=HistogramOverlayProps());
	virtual ~HistogramOverlay() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool processEOS(string& pinId);
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();	
	bool validateInputOutputPins();
	bool shouldTriggerSOS();

private:	
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};



