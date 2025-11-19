#include "QuePushStrategy.h"
#include "FrameContainerQueue.h"
#include "Logger.h"

std::shared_ptr<QuePushStrategy> QuePushStrategy::getStrategy(QuePushStrategyType type, std::string& srcModuleId)
{
	switch (type)
	{
		case QuePushStrategy::NON_BLOCKING_ALL_OR_NONE:
			return std::make_shared<NonBlockingAllOrNonePushStrategy>(srcModuleId);
		case QuePushStrategy::NON_BLOCKING_ANY:
			return std::make_shared<NonBlockingAnyPushStrategy>(srcModuleId);
		default:
			return std::make_shared<QuePushStrategy>(srcModuleId);
	}
}

QuePushStrategy::QuePushStrategy(std::string& srcModuleId): mId(srcModuleId)
{

}

QuePushStrategy::~QuePushStrategy()
{
	mQueByModule.clear();
}

void QuePushStrategy::addQue(std::string dstModuleId, std::shared_ptr<FrameContainerQueue>& que)
{
	mQueByModule[dstModuleId] = que;
}

void QuePushStrategy::push(std::string dstModuleId, frame_container& frames)
{
	mQueByModule[dstModuleId]->push(frames);
}

NonBlockingAnyPushStrategy::NonBlockingAnyPushStrategy(std::string& srcModuleId) : QuePushStrategy(srcModuleId), mDropCount(0), mPrintFrequency(1000)
{

}

NonBlockingAnyPushStrategy::~NonBlockingAnyPushStrategy()
{

}

void NonBlockingAnyPushStrategy::push(std::string dstModuleId, frame_container& frames)
{
	auto ret = mQueByModule[dstModuleId]->try_push(frames);
	if (!ret)
	{
		mDropCount++;
		if (mDropCount%mPrintFrequency == 1)
		{
			// LOG_ERROR << mId << "<" << dstModuleId << "> dropping from que. DropCount<" << mDropCount << ">";
		}		
	}
}

NonBlockingAllOrNonePushStrategy::NonBlockingAllOrNonePushStrategy(std::string& srcModuleId) : QuePushStrategy(srcModuleId)
{

}

NonBlockingAllOrNonePushStrategy::~NonBlockingAllOrNonePushStrategy()
{
	mFramesByModule.clear();
}

void NonBlockingAllOrNonePushStrategy::push(std::string dstModuleId, frame_container& frames)
{
	auto ret = mFramesByModule.insert({dstModuleId, frames});
	if (!ret.second)
	{
		LOG_ERROR << mId << "<" << dstModuleId << "> already has an entry. Not expected to come here.";
	}
}

bool NonBlockingAllOrNonePushStrategy::flush()
{
	bool allQuesFree = true;
	uint32_t lockCounter = 0;
	std::string firstFullModuleId;
	for (const auto& [moduleId, frames] : mFramesByModule)
	{
		auto& que = mQueByModule[moduleId];
		que->acquireLock();
		lockCounter++;

		if (!que->is_not_full())
		{
			allQuesFree = false;
			firstFullModuleId = moduleId;
			break;
		}
	}


	if (allQuesFree)
	{
		for (const auto& [moduleId, frames] : mFramesByModule)
		{
			mQueByModule[moduleId]->pushUnsafeForQuePushStrategy(frames);
		}
	}
	else
	{

		for (const auto& [moduleId, frames] : mFramesByModule)
		{
			mQueByModule[moduleId]->releaseLock();
			lockCounter--;
			if (lockCounter == 0)
			{
				break;
			}
		}
		// LOG_ERROR << mId << "<> skipping all from que because <" << firstFullModuleId << "> is full";
	}

	mFramesByModule.clear();
	return true;
}