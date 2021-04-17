#pragma once

#include "Module.h"
#include <memory>

class DMAFDToHostCopyProps : public ModuleProps
{
public:
	DMAFDToHostCopyProps() : ModuleProps()
	{
	}
};

class DMAFDToHostCopy : public Module
{
public:
	DMAFDToHostCopy(DMAFDToHostCopyProps props = DMAFDToHostCopyProps());
	virtual ~DMAFDToHostCopy();

	virtual bool init();
	virtual bool term();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool processEOS(string &pinId);

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
