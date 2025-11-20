#pragma once

#include "Module.h"

class VirtualCameraSinkProps : public ModuleProps
{
public:
	VirtualCameraSinkProps(std::string _device) : device(_device)
	{
	}

	std::string device;
};

class VirtualCameraSink : public Module
{

public:
	VirtualCameraSink(VirtualCameraSinkProps _props);
	virtual ~VirtualCameraSink();
	bool init() override;
	bool term() override;

	void getImageSize(int &width, int &height);

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool shouldTriggerSOS() override;
	bool processEOS(string &pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
