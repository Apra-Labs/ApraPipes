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

	bool init();
	bool term();
	bool setPausedState(bool state);
	bool pausedState = false;
	bool keyFrameAfterPause = false;
	boost::shared_ptr<Frame> savedIFrame;
	boost::thread pauserThread;
	void pauserThreadFunction();
	void setFps(int fps);
protected:
	bool process(frame_container &frames);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processSOS(frame_sp &frame);
	bool processEOS(string& pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	int sleepTimeInMilliSec;
	using sys_clock = std::chrono::system_clock;
	sys_clock::time_point frame_begin;
	std::chrono::nanoseconds myNextWait;
	std::chrono::nanoseconds myTargetFrameLen;
	bool initDone = false;
	int fps;
	void start();
	void end();
};
