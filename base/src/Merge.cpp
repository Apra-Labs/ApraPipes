#include <boost/foreach.hpp>

#include "Merge.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"

class Merge::Detail
{

public:
	Detail(MergeProps& _props):maxDelay(_props.maxDelay), lastIndex(0), mQueueSize(0)
	{

	}

	~Detail() 
	{
		clear();
	}

	void setOutputPinId(std::string& pinId)
	{
		mOutputPinId = pinId;
	}
		
	bool queue(frame_container& frames)
	{		
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			auto& frame = it->second;
			if (frame->fIndex2 < lastIndex)
			{
				// frame index jumped may be because of drop upstream
				continue;
			}
			mQueue[frame->fIndex2] = frame;	
			mQueueSize++;
		}			

		return true;
	}

	bool get(frame_container& frames)
	{
		auto elem = mQueue.begin();

		if (mQueueSize > maxDelay)
		{
			// this should flush
			lastIndex = elem->first - 1;
		}			

		if (mQueueSize == 0 || (elem->first != (lastIndex + 1) && !(elem->first == 0 && lastIndex == 0) ) )
		{
			return false;
		}

		frames[mOutputPinId] = elem->second;
		removeElement(elem->first);

		return true;
	}

	void clear()
	{		
		mQueue.clear();
	}
	
private:		
	void removeElement(uint64_t index)
	{
		lastIndex = index;
		mQueue.erase(index);
		mQueueSize--;
	}

	uint32_t mQueueSize;
	uint32_t maxDelay;
	uint64_t lastIndex;
	std::map<uint64_t, frame_sp> mQueue;
	std::string mOutputPinId;
};


Merge::Merge(MergeProps _props):Module(TRANSFORM, "Merge", _props)
{
	mDetail.reset(new Detail(_props));
}

bool Merge::validateInputPins()
{	
	auto frameType = -1;
	for (auto const& elem: getInputMetadata())
	{
		if (frameType == -1)
		{
			frameType = elem.second->getFrameType();
			continue;
		}

		if (frameType == elem.second->getFrameType())
		{
			continue;
		}

		LOG_ERROR << "All inputs must be of same type. Expected<" << frameType << "> Actual<" << elem.second->getFrameType() << ">";
		return false;
	}

	return true;
}

bool Merge::validateOutputPins()
{
	if(getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "Expected only 1 output pin. But Actual<>" << getNumberOfOutputPins();
		return false;
	}

	return true;
}

bool Merge::init()
{
	if (!Module::init())
	{
		return false;
	}
	
	
	return true;
}

bool Merge::term()
{
	mDetail.reset();
	return Module::term();
}

void Merge::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	if(getNumberOfOutputPins() == 0)
    {        
       auto pinId = addOutputPin(metadata);
	   mDetail->setOutputPinId(pinId);
    }       
}

bool Merge::process(frame_container& frames)
{		
	mDetail->queue(frames);
	
	frame_container outFrames;
	while (mDetail->get(outFrames))
	{
		send(outFrames);
	}

	return true;
}