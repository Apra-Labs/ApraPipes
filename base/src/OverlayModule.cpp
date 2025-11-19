#include <cstdint>
#include "OverlayModule.h"
#include "Utils.h"
#include "FrameContainerQueue.h"
#include "Overlay.h"

OverlayModule::OverlayModule(OverlayModuleProps _props) : Module(TRANSFORM, "OverlayModule", _props) {}

void OverlayModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
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
	auto inputMetadataByPin = getInputMetadata();
	for (const auto& me : inputMetadataByPin)
	{
		FrameMetadata::FrameType frameType = me.second->getFrameType();
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE OR OVERLAY_INFO_IMAGE. Actual<" << frameType << ">";
			return false;
		}
	}
	return true;
}

bool OverlayModule::validateOutputPins()
{
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
	DrawingOverlay drawOverlay;
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto metadata = it->second->getMetadata();
		auto frameType = metadata->getFrameType();
		frame_sp frame = it->second;

		if (frameType == FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			drawOverlay.deserialize(frame);
		}

		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			drawOverlay.draw(frame);
			frame_container overlayConatiner;
			overlayConatiner.insert(make_pair(mOutputPinId, frame));
			send(overlayConatiner);
		}
	}
	return true;
}
