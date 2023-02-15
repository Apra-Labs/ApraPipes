#pragma once
#include <string>
#include "Module.h"

using namespace std;

class MotionExtractorProps : public ModuleProps
{
public:
	MotionExtractorProps()
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
};

class MotionExtractor : public Module
{
public:
	MotionExtractor(MotionExtractorProps _props);
	virtual ~MotionExtractor() {};
	bool init();
	bool term();
protected:
	bool process(frame_container& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	std::string mOutputPinId;
};
