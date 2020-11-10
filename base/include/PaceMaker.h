#pragma once
#include <chrono>
#include <thread>

class PaceMaker {
	using sys_clock = std::chrono::system_clock;
	sys_clock::time_point frame_begin;
	std::chrono::nanoseconds myNextWait;
	std::chrono::nanoseconds myTargetFrameLen;
	bool initDone = false;
	int fps;
	
public:	

	PaceMaker(int _fps)
	{		
		setFps(_fps);
	}

	void setFps(int _fps)
	{
		if(_fps <= 0)
		{
			myTargetFrameLen = std::chrono::nanoseconds(1);
		}
		else 
		{
			myTargetFrameLen = std::chrono::nanoseconds(1000000000 / _fps);
		}
		fps = _fps;
	}

	void start() {
		if (!initDone)
		{
			myNextWait = myTargetFrameLen;
			frame_begin = sys_clock::now();
			initDone = true;
		}
	}

	void end() {
		
		std::chrono::nanoseconds frame_len = sys_clock::now() - frame_begin;
		if (myNextWait > frame_len)
		{
			std::this_thread::sleep_for(myNextWait - frame_len);
		}
		myNextWait += myTargetFrameLen;
	}

};