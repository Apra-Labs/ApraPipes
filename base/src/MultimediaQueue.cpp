#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <stdafx.h>
#include <map>
#include "Frame.h"
#include "MultimediaQueue.h"
#include "Logger.h"

class QueueClass
{

public:
	~QueueClass()
	{}

	bool enqueue(frame_container& frames, uint32_t lowerWaterMark, uint32_t upperWaterMark, bool isMapDelayInTime, bool  pushNext)
	{	//	Here the frame_containers are inserted into the map
		uint64_t largestTimeStamp = 0;
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			mQueue.insert({ it->second->timestamp, frames });
			if (largestTimeStamp < it->second->timestamp)
			{
				largestTimeStamp = it->second->timestamp;
			}
		}
		if (isMapDelayInTime) // If the lower and upper watermark are given in time
		{
			if ((largestTimeStamp - mQueue.begin()->first > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((largestTimeStamp - mQueue.begin()->first > upperWaterMark) && (pushNext == false))
			{
				auto it = mQueue.begin();
				auto lastElement = mQueue.end();
				lastElement--;
				auto lastElementTimeStamp = lastElement->first;
				while (it != mQueue.end())
				{
					if ((lastElementTimeStamp - it->first) < lowerWaterMark)
					{
						break;
					}
					auto itr = it;
					++it;
					mQueue.erase(itr->first);
				}
				pushNext = true;
			};
		}
		else // If the lower and upper water mark are given in number of frames
		{
			if ((mQueue.size() > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((mQueue.size() > upperWaterMark) && (pushNext == false))
			{
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{
					if (mQueue.size() < lowerWaterMark)
					{
						break;
					}
					auto itr = it;
					++it;
					mQueue.erase(itr->first);
				}
				pushNext = true;
			};
		}
		return true;
	}
	typedef std::map<uint64_t, frame_container> MultimediaQueueMap;
	MultimediaQueueMap mQueue;
};

//State Design begins here
class Export : public State {
public:
	Export() : State(StateType::EXPORT) {}
	Export(boost::shared_ptr<QueueClass> queueObj) : State(StateType::EXPORT) {
		queueObject = queueObj;
	}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap) override
	{
		auto tOld = queueMap.begin()->first;
		auto temp = queueMap.end();
		temp--;
		auto tNew = temp->first;
		queryEnd = queryEnd - 1;
		if ((queryStart < tOld) && (queueMap.upper_bound(queryEnd) != queueMap.end()))
		{
			queryStart = tOld;
			timeReset = true;
			return true;
		}
		else if ((queryEnd > tNew) && (queueMap.upper_bound(queryStart) != queueMap.end()))
		{
			if (tNew >= queryEnd)
			{
				timeReset = true;
			}
			queryEnd = tNew;

			return true;
		}
		else
		{
			timeReset = true;
			return true;
		}
	}
};

class Waiting : public State {
public:
	Waiting() : State(State::StateType::WAITING) {}
	Waiting(boost::shared_ptr<QueueClass> queueObj) : State(State::StateType::WAITING) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap) override
	{
		BOOST_LOG_TRIVIAL(info) << "WAITING STATE : THE FRAMES ARE IN FUTURE!! WE ARE WAITING FOR THEM..";
		return true;
	}
};

class Idle : public State {
public:
	Idle() : State(StateType::IDLE) {}
	Idle(boost::shared_ptr<QueueClass> queueObj) : State(StateType::IDLE) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap) override
	{
		//The code will not come here
		return true;
	}
};

MultimediaQueue::MultimediaQueue(MultimediaQueueProps _props) :Module(TRANSFORM, "MultimediaQueue", _props), mProps(_props)
{
	mState.reset(new State());
	mState->queueObject.reset(new QueueClass());
}

bool MultimediaQueue::validateInputPins()
{
	return true;
}

bool MultimediaQueue::validateOutputPins()
{
	return true;
}

bool MultimediaQueue::validateInputOutputPins()
{
	return Module::validateInputOutputPins();
}

bool MultimediaQueue::setNext(boost::shared_ptr<Module> next, bool open, bool sieve)
{
	return Module::setNext(next, open, false, sieve);
}

bool MultimediaQueue::init()
{
	if (!Module::init())
	{
		return false;
	}

	mState.reset(new Idle(mState->queueObject));

	return true;
}

bool MultimediaQueue::term()
{
	mState.reset();
	return Module::term();
}

void MultimediaQueue::getQueueBoundaryTS(uint64_t& tOld, uint64_t& tNew)
{
	auto queueMap = mState->queueObject->mQueue;
	tOld = queueMap.begin()->first;
	auto tempIT = queueMap.end();
	tempIT--;
	tNew = tempIT->first;
}

void MultimediaQueue::setState(uint64_t tStart, uint64_t tEnd)
{
	uint64_t tOld, tNew = 0;
	getQueueBoundaryTS(tOld, tNew);

	//Checking conditions to determine the new state and transitions to it.

	if (tEnd < tOld)
	{
		BOOST_LOG_TRIVIAL(info) << "IDLE STATE : MAYBE THE FRAMES HAVE PASSED THE QUEUE";
		mState.reset(new Idle(mState->queueObject));
	}
	else if (tStart > tNew)
	{
		mState.reset(new Waiting(mState->queueObject));
	}
	else
	{
		mState.reset(new Export(mState->queueObject));
	}

}

bool MultimediaQueue::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::MultimediaQueue)
	{
		MultimediaQueueCommand cmd;
		getCommand(cmd, frame);
		setState(cmd.startTime, cmd.endTime);
		queryStartTime = cmd.startTime;
		startTimeSaved = cmd.startTime;
		queryEndTime = cmd.endTime;
		endTimeSaved = cmd.endTime;

		bool reset = false;
		bool pushNext = true;
		if (mState->Type == State::EXPORT)
		{
			mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue);
			for (auto it = mState->queueObject->mQueue.begin(); it != mState->queueObject->mQueue.end(); it++)
			{
				if (((it->first) >= queryStartTime) && (((it->first) <= queryEndTime)))
				{
					if (isNextModuleQueFull())
					{
						pushNext = false;
						break;
					}
					else
					{
						send(it->second);
					}
				}
			}
		}

		if (mState->Type == mState->EXPORT)
		{
			uint64_t tOld = 0, tNew = 0;
			getQueueBoundaryTS(tOld, tNew);
			if (endTimeSaved > tNew)
			{
				reset = false;
			}
			queryStartTime = tNew;
		}
		if (reset)
		{
			queryStartTime = 0;
			queryEndTime = 0;
			setState(queryStartTime, queryEndTime);
		}
		return true;
	}
}

bool MultimediaQueue::allowFrames(uint64_t& ts, uint64_t& te)
{
	if (mState->Type != mState->EXPORT)
	{
		MultimediaQueueCommand cmd;
		cmd.startTime = ts;
		cmd.endTime = te;
		return queueCommand(cmd);
	};
	return true;
}

bool MultimediaQueue::process(frame_container& frames)
{
	mState->queueObject->enqueue(frames, mProps.lowerWaterMark, mProps.upperWaterMark, mProps.isMapDelayInTime, pushNext);
	if (mState->Type == State::EXPORT)
	{
		uint64_t tOld, tNew = 0;
		getQueueBoundaryTS(tOld, tNew);
		queryEndTime = tNew;
	}

	if (mState->Type == State::WAITING)
	{
		setState(queryStartTime, queryEndTime);
	}
	
	if (mState->Type == State::EXPORT)
	{
		mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue);
		for (auto it = mState->queueObject->mQueue.begin(); it != mState->queueObject->mQueue.end(); it++)
		{
			if (((it->first) >= (queryStartTime + 1)) && (((it->first) <= (endTimeSaved))))
			{
				if (isNextModuleQueFull())
				{
					pushNext = false;
				}
				else
				{
					send(it->second);
				}
			}
		}
	}
	if (mState->Type == State::EXPORT)
	{
		uint64_t tOld, tNew = 0;
		getQueueBoundaryTS(tOld, tNew);
		if (queryEndTime > tNew);
		{
			reset = false;
		}
		queryStartTime = tNew;
	}

	if (reset)
	{
		queryStartTime = 0;
		queryEndTime = 0;
		setState(queryStartTime, queryEndTime);
	}
	return true;
}

bool MultimediaQueue::handlePropsChange(frame_sp& frame)
{
	if (mState->Type != State::EXPORT)
	{
		MultimediaQueueProps props(10, 5, false);
		auto ret = Module::handlePropsChange(frame, props);
		return ret;
	}

	else
	{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
		return true;
	}
}

MultimediaQueueProps MultimediaQueue::getProps()
{
	fillProps(mProps);
	return mProps;
}

void  MultimediaQueue::setProps(MultimediaQueueProps _props)
{
	if (mState->Type != State::EXPORT)
	{
		mProps = _props;
		Module::addPropsToQueue(mProps);
	}
	else
	{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
	}
}