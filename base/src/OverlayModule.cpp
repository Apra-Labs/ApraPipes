#include <cstdint>
#include <boost/foreach.hpp>
#include "OverlayModule.h"
#include "Utils.h"
#include "FrameContainerQueue.h"
#include "Overlay.h"

OverlayModule::OverlayModule(OverlayModuleProps _props) : Module(TRANSFORM, "OverlayModule", _props) {}

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
	/*if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}*/
	return true;
}

bool OverlayModule::validateOutputPins()
{
	/*if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}*/
	return true;
}

bool OverlayModule::shouldTriggerSOS()
{
	return true;
}

bool OverlayModule::process(frame_container& frames)
{
	DrawingOverlay component;
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto metadata = it->second->getMetadata();
		auto frameType = metadata->getFrameType();
		frame_sp frame = it->second;

		if (frameType == FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			component.deserialize(frame);
		}

		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			component.mDraw(frame);
			frame_container overlayConatiner;
			overlayConatiner.insert(make_pair(mOutputPinId, frame));
			send(overlayConatiner);
		}
	}
	return true;
}
