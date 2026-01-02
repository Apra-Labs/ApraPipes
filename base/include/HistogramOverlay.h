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

	bool init() override;
	bool term() override;

protected:
	bool processEOS(string& pinId) override;
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool validateInputOutputPins() override;
	bool shouldTriggerSOS() override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};



