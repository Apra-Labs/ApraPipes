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
	pair<string, framemetadata_sp> me; // map element	
	auto inputMetadataByPin = getInputMetadata();
	BOOST_FOREACH(me, inputMetadataByPin)
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
	std::cout << "[OverlayModule] process() called" << std::endl;

	DrawingOverlay drawOverlay;

	std::cout << "First pinId in frames : " << frames.cbegin()->first << std::endl;
	std::cout << "First frame size: " << frames.cbegin()->second->size() << std::endl;


	if (!frames.empty())
	{
		auto last = std::prev(frames.cend());
		std::cout << "Last pinId: " << last->first << std::endl;
		std::cout << "Last frame size: " << last->second->size() << std::endl;
	}



	for (auto it = frames.cbegin(); it != frames.cend(); ++it)
	{


		std::cout << "[OverlayModule] check point 1: iterating over input frames" << std::endl;

		auto metadata = it->second->getMetadata();
		auto frameType = metadata->getFrameType();

		std::cout << "[OverlayModule] frame received with type = " << frameType << std::endl;

		frame_sp frame = it->second;

		if (frameType == FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			std::cout << "[OverlayModule] Deserializing OVERLAY_INFO_IMAGE" << std::endl;
			drawOverlay.deserialize(frame);
		}
		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			std::cout << "[OverlayModule] Drawing on RAW_IMAGE" << std::endl;
			drawOverlay.draw(frame);

			frame_container overlayContainer;
			overlayContainer.insert(make_pair(mOutputPinId, frame));

			std::cout << "[OverlayModule] Sending processed frame downstream" << std::endl;
			send(overlayContainer);
		}
		else
		{
			std::cout << "[OverlayModule] Unknown frame type received: " << frameType << std::endl;
		}
	}

	std::cout << "[OverlayModule] process() completed" << std::endl;

	return true;
}

