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

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, string &pinId) override; // throws exception if validation fails
	bool processEOS(string &pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
