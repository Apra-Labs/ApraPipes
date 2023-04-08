#include <cstdint>
#include <boost/foreach.hpp>
#include "OverlayModule.h"
#include "Utils.h"
#include "FrameContainerQueue.h"
#include "OverlayDataInfo.h"

enum OverlayType
{
	LINE = 0,
	RECTANGLE,
	CIRCLE,
	NOT_KNOWN
};

class OverlayComponent
{
public:
	OverlayComponent() {}
	class OverlayComponent(OverlayType& _shapeType, frame_sp inFrame)
	{
		
	}
	virtual void draw(frame_sp inRawFrame) = 0;
};

class Line : public OverlayComponent
{
public:

	void draw(frame_sp inRawFrame)
	{
		printf("draw line");
	}
};

class Circle : public OverlayComponent
{
public:
	void draw(frame_sp inRawFrame)
	{
		printf("draw circle");
	}
};

class rectangle : public OverlayComponent
{
public:
	rectangle(OverlayType& enumType, frame_sp frame)
	{

	}
	void draw(frame_sp inRawFrame)
	{
		printf("draw Rectangle");
	}
};

class Composite : public OverlayComponent
{
public:
	void add(OverlayComponent* componentObj)
	{
		gList.push_back(componentObj);
	}
	void draw(frame_sp inRawFrame)
	{
		printf("draw compisite");
	}

private:
	vector<OverlayComponent*> gList;
};

class OverlayCommand
{
public:
	OverlayCommand() {}
	
	virtual void execute(OverlayComponent* Overlay, frame_sp inRawFrame)
	{

		Overlay->draw(inRawFrame);
	}
public:
	boost::shared_ptr<FrameContainerQueueOverlayAdapter> frameContainerOverlayAdapt;;
	Line* lineObj;
};


OverlayModule::OverlayModule(OverlayModuleProps props) : Module(TRANSFORM, "OverlayMotionVectors", props)
{
	mDetail.reset(new OverlayCommand());
	mDetail->frameContainerOverlayAdapt.reset(new FrameContainerQueueOverlayAdapter([&](size_t size) -> frame_sp {return makeFrame(size); }));
}

void OverlayModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	if(metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	mOutputPinId = addOutputPin(metadata);
}


bool OverlayModule::init()
{
	return Module::init();
}

bool OverlayModule::term()
{
	return Module::term();
}

bool OverlayModule::validateInputPins()
{
	if (getNumberOfInputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	pair<string, framemetadata_sp> me; // map element	
	auto inputMetadataByPin = getInputMetadata();
	BOOST_FOREACH(me, inputMetadataByPin)
	{
		FrameMetadata::FrameType frameType = me.second->getFrameType();
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::MOTION_VECTOR_DATA)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE OR MOTION_VECTOR_DATA. Actual<" << frameType << ">";
			return false;
		}
	}
	return true;
}

bool OverlayModule::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	auto outputMetadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = outputMetadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool OverlayModule::shouldTriggerSOS()
{
	return true;
}

bool OverlayModule::process(frame_container& frames)
{
	frame_sp outFrame;
	OverlayComponent* OverlayObj;
	if (frames.begin()->second->mFrameType == FrameMetadata::FrameType::RAW_IMAGE)
	mDetail->execute(OverlayObj, frames.begin()->second);
	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);
	return true;
}

bool OverlayModule::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	return true;
}