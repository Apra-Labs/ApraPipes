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
	bool init();
	bool term();

	void getImageSize(int &width, int &height);

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
