#include <boost/foreach.hpp>

#include "Split.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"

Split::Split(SplitProps _props):Module(TRANSFORM, "Split", _props), mNumber(_props.number), mCurrentIndex(0), mFIndex2(0)
{
	
}

bool Split::validateInputPins()
{	
	if(getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "Expected only 1 input pins. But Actual<>" << getNumberOfInputPins();
		return false;
	}

	return true;	
}

bool Split::validateOutputPins()
{
	if (getNumberOfOutputPins() > mNumber)
	{
		LOG_ERROR << "Expected <" << mNumber << ">. But Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

bool Split::init()
{
	if (!Module::init())
	{
		return false;
	}	
	
	return true;
}

bool Split::term()
{	
	return Module::term();
}

void Split::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	for(uint32_t i = 0; i < mNumber; i++)
	{
		mPinIds.push_back(addOutputPin(metadata));
	}
}

bool Split::process(frame_container& frames)
{		
	auto frame = frames.begin()->second;
	frame->fIndex2 = mFIndex2++;
	frames.insert(std::make_pair(mPinIds[mCurrentIndex], frame));
	
	send(frames);
	mCurrentIndex++;
	if(mCurrentIndex == mNumber)
	{
		mCurrentIndex = 0;
	}

	return true;
}