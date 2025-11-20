#pragma once

#include "Module.h"

class RTSPPusherProps : public ModuleProps
{
public:
	RTSPPusherProps(std::string _URL, std::string _title) : URL(_URL), title(_title), isTCP(true), encoderTargetKbps(2*1024)
	{
	}

	~RTSPPusherProps()
	{
	}

	std::string URL;
	std::string title;
	bool isTCP;
	uint32_t encoderTargetKbps;
};

class RTSPPusher : public Module
{
public:
	enum EventType
	{
		CONNECTION_READY = 1,
		CONNECTION_FAILED = 2,
		WRITE_FAILED,
		STREAM_ENDED
	};

	RTSPPusher(RTSPPusherProps props);
	virtual ~RTSPPusher();

	bool init() override;
	bool term() override;

protected:
	bool process(frame_container &frames) override;
	bool validateInputPins() override;
	bool shouldTriggerSOS() override;
	bool processSOS(frame_sp &frame) override;
	bool processEOS(string& pinId) override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};
