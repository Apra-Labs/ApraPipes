#include <boost/foreach.hpp>

#include "FramesMuxer.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"

class FramesMuxerStrategy
{

public:
	FramesMuxerStrategy(FramesMuxerProps& _props) {}

	virtual ~FramesMuxerStrategy()
	{

	}

	virtual std::string addInputPin(std::string& pinId)
	{
		return getMuxOutputPinId(pinId);
	}

	virtual bool queue(frame_container& frames)
	{

		return true;
	}

	virtual bool get(frame_container& frames)
	{
		return false;
	}

protected:
	std::string getMuxOutputPinId(const std::string& pinId)
	{
		return pinId + "_mux_";
	}
};


class AllOrNoneStrategy : public FramesMuxerStrategy
{

public:
	AllOrNoneStrategy(FramesMuxerProps& _props) :FramesMuxerStrategy(_props), maxDelay(_props.maxDelay) {}

	~AllOrNoneStrategy()
	{
		clear();
	}

	std::string addInputPin(std::string& pinId)
	{
		mQueue[pinId] = boost::container::deque<frame_sp>();

		return FramesMuxerStrategy::addInputPin(pinId);
	}

	bool queue(frame_container& frames)
	{
		// add all the frames to the que
		// store the most recent fIndex
		size_t fIndex = 0;
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			mQueue[it->first].push_back(it->second);
			if (fIndex < it->second->fIndex)
			{
				fIndex = it->second->fIndex;
			}
		}

		// loop over the que first frames and store the first highest
		size_t firstHighest = 0;
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			if (frames_arr.size())
			{
				auto& frame = frames_arr.front();
				if (firstHighest < frame->fIndex)
				{
					firstHighest = frame->fIndex;
				}
			}
		}

		// loop over the que and remove old frames using maxDelay
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			while (frames_arr.size())
			{
				auto& frame = frames_arr.front();
				if (frame->fIndex < firstHighest || fIndex - frame->fIndex > maxDelay)
				{
					frames_arr.pop_front();
				}
				else
				{
					break;
				}
			}
		}

		return true;
	}

	bool get(frame_container& frames)
	{
		bool allFound = true;
		size_t fIndex = 0;
		bool firstIter = true;

		//check the first in each queue
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			if (frames_arr.size() == 0)
			{
				allFound = false;
				break;
			}

			auto& frame = frames_arr.front();
			if (firstIter)
			{
				firstIter = false;
				fIndex = frame->fIndex;
			}

			if (fIndex != frame->fIndex)
			{
				allFound = false;
				break;
			}
		}

		if (!allFound)
		{
			return false;
		}

		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			frames[FramesMuxerStrategy::getMuxOutputPinId(it->first)] = it->second.front();
			it->second.pop_front();
		}

		return true;
	}

	void clear()
	{
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			it->second.clear();
		}
		mQueue.clear();
	}

private:

	typedef std::map<std::string, boost_deque<frame_sp>> MuxerQueue; // pinId and frame
	MuxerQueue mQueue;

	int maxDelay;

};




class MaxDelayAnyStrategy : public FramesMuxerStrategy
{

	/*
	 * waits for the delay number of frames
	 * after reaching the max delay - send the available frames
	 */

public:
	MaxDelayAnyStrategy(FramesMuxerProps& _props) :FramesMuxerStrategy(_props), maxDelay(_props.maxDelay), mLastFrameIndex(0) {}

	~MaxDelayAnyStrategy()
	{
		clear();
	}

	std::string addInputPin(std::string& pinId)
	{
		mQueue[pinId] = boost::container::deque<frame_sp>();

		return FramesMuxerStrategy::addInputPin(pinId);
	}

	bool queue(frame_container& frames)
	{
		// add all the frames to the que
		// store the most recent fIndex
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			mQueue[it->first].push_back(it->second);
			if (mLastFrameIndex < it->second->fIndex)
			{
				mLastFrameIndex = it->second->fIndex;
			}
		}


		return true;
	}

	bool get(frame_container& frames)
	{
		bool oldFramesAvailable = false;

		// loop over the que first frames and store the first highest
		size_t firstHighest = 0;
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			if (frames_arr.size())
			{
				auto& frame = frames_arr.front();
				if (firstHighest < frame->fIndex)
				{
					firstHighest = frame->fIndex;
				}
			}
		}

		// loop over the que and send old frames using maxDelay
		size_t selectedFrameIndex = 0;
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			auto queLen = frames_arr.size();
			if (queLen)
			{
				auto& frame = frames_arr.front();

				if (frame->fIndex < firstHighest || mLastFrameIndex - frame->fIndex > maxDelay) // condition to check if the frame is old
				{
					frames[FramesMuxerStrategy::getMuxOutputPinId(it->first)] = frame;
					oldFramesAvailable = true;
					selectedFrameIndex = frame->fIndex; // found 1 old frame  
					break;
				}
			}
		}

		if (oldFramesAvailable)
		{
			for (auto it = mQueue.begin(); it != mQueue.end(); it++)
			{
				auto& frames_arr = it->second;
				if (frames_arr.size())
				{
					auto& frame = frames_arr.front();
					// send all frames matching the index together
					if (frame->fIndex == selectedFrameIndex)
					{
						frames[FramesMuxerStrategy::getMuxOutputPinId(it->first)] = frame;
						frames_arr.pop_front();
					}
				}
			}

			return true;
		}

		bool allFound = true;
		size_t fIndex = 0;
		bool firstIter = true;

		//check the first in each queue
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			auto& frames_arr = it->second;
			if (frames_arr.size() == 0)
			{
				allFound = false;
				break;
			}

			auto& frame = frames_arr.front();
			if (firstIter)
			{
				firstIter = false;
				fIndex = frame->fIndex;
			}

			if (fIndex != frame->fIndex)
			{
				allFound = false;
				break;
			}
		}

		if (allFound)
		{

			for (auto it = mQueue.begin(); it != mQueue.end(); it++)
			{
				frames[FramesMuxerStrategy::getMuxOutputPinId(it->first)] = it->second.front();
				it->second.pop_front();
			}
		}

		return allFound;
	}

	void clear()
	{
		for (auto it = mQueue.begin(); it != mQueue.end(); it++)
		{
			it->second.clear();
		}
		mQueue.clear();
	}

private:

	typedef std::map<std::string, boost_deque<frame_sp>> MuxerQueue; // pinId and frame
	MuxerQueue mQueue;

	int maxDelay;
	size_t mLastFrameIndex;

};


FramesMuxer::FramesMuxer(FramesMuxerProps _props) :Module(TRANSFORM, "FramesMuxer", _props)
{

	switch (_props.strategy)
	{
	case FramesMuxerProps::ALL_OR_NONE:
		mDetail.reset(new AllOrNoneStrategy(_props));
		break;
	case FramesMuxerProps::MAX_DELAY_ANY:
		mDetail.reset(new MaxDelayAnyStrategy(_props));
		break;
	default:
		LOG_ERROR << "Strategy not implemented " << _props.strategy;
		break;
	}
}

bool FramesMuxer::validateInputPins()
{
	return true;
}

bool FramesMuxer::validateOutputPins()
{
	return true;
}

bool FramesMuxer::validateInputOutputPins()
{
	if (getNumberOfInputPins() < 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2 or more. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return Module::validateInputOutputPins();
}

bool FramesMuxer::init()
{
	if (!Module::init())
	{
		return false;
	}


	return true;
}

bool FramesMuxer::term()
{
	mDetail.reset();
	return Module::term();
}

void FramesMuxer::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto outputPinId = mDetail->addInputPin(pinId);
	addOutputPin(metadata, outputPinId);
}

bool FramesMuxer::process(frame_container& frames)
{
	mDetail->queue(frames);

	frame_container outFrames;
	while (mDetail->get(outFrames))
	{
		send(outFrames);
		outFrames.clear();
	}

	return true;
}