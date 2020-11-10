#include "QuePushStrategy.h"
#include "FrameContainerQueue.h"
#include "Logger.h"

boost::shared_ptr<QuePushStrategy> QuePushStrategy::getStrategy(QuePushStrategyType type, std::string& srcModuleId)
{
	switch (type)
	{
		case QuePushStrategy::NON_BLOCKING_ALL_OR_NONE:	
			return boost::shared_ptr<QuePushStrategy>(new NonBlockingAllOrNonePushStrategy(srcModuleId));
		case QuePushStrategy::NON_BLOCKING_ANY:		
			return boost::shared_ptr<QuePushStrategy>(new NonBlockingAnyPushStrategy(srcModuleId));
		default:
			return boost::shared_ptr<QuePushStrategy>(new QuePushStrategy(srcModuleId));
	}
}

QuePushStrategy::QuePushStrategy(std::string& srcModuleId): mId(srcModuleId)
{

}

QuePushStrategy::~QuePushStrategy()
{
	mQueByModule.clear();
}

void QuePushStrategy::addQue(std::string dstModuleId, boost::shared_ptr<FrameContainerQueue>& que)
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
	auto ret = mFramesByModule.insert(std::make_pair(dstModuleId, frames));
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
	for (frames_by_module::const_iterator it = mFramesByModule.cbegin(); it != mFramesByModule.cend(); ++it)
	{
		auto& que = mQueByModule[it->first];
		que->acquireLock();
		lockCounter++;

		if (!que->is_not_full())
		{
			allQuesFree = false;	
			firstFullModuleId = it->first;
			break;
		}			
	}


	if (allQuesFree)
	{
		for (frames_by_module::const_iterator it = mFramesByModule.cbegin(); it != mFramesByModule.cend(); ++it)
		{
			mQueByModule[it->first]->pushUnsafeForQuePushStrategy(it->second);
		}
	}	
	else
	{
		
		for (frames_by_module::const_iterator it = mFramesByModule.cbegin(); it != mFramesByModule.cend(); ++it)
		{
			mQueByModule[it->first]->releaseLock();
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