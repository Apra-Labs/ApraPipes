#pragma once

#include "CommonDefs.h"

class Module;
class FrameContainerQueue;

class QuePushStrategy
{
public:
	enum QuePushStrategyType {		
		BLOCKING,
		NON_BLOCKING_ANY,
		NON_BLOCKING_ALL_OR_NONE
	};

	static boost::shared_ptr<QuePushStrategy> getStrategy(QuePushStrategyType type, std::string& srcModuleId);

	QuePushStrategy(std::string& srcModuleId);
	virtual ~QuePushStrategy();

	void addQue(std::string dstModuleId, boost::shared_ptr<FrameContainerQueue>& que);

	virtual void push(std::string dstModuleId, frame_container& frames);
	virtual bool flush() { return true; }

protected:
	std::map<std::string, boost::shared_ptr<FrameContainerQueue>> mQueByModule;
	std::string mId;
};

class NonBlockingAnyPushStrategy : public QuePushStrategy
{
public:
	NonBlockingAnyPushStrategy(std::string& srcModuleId);
	virtual ~NonBlockingAnyPushStrategy();

	virtual void push(std::string dstModuleId, frame_container& frames);
private:	
	uint64_t mDropCount;
	uint64_t mPrintFrequency;
};

class NonBlockingAllOrNonePushStrategy: public QuePushStrategy
{
public:
	NonBlockingAllOrNonePushStrategy(std::string& srcModuleId);
	virtual ~NonBlockingAllOrNonePushStrategy();

	virtual void push(std::string dstModuleId, frame_container& frames);
	virtual bool flush();

private:
	typedef std::map<std::string, frame_container> frames_by_module;
	frames_by_module mFramesByModule;
};