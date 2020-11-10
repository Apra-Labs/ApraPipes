#pragma once

#include "Module.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "ApraData.h"

class ExternalSourceModuleProps : public ModuleProps
{
public:
	ExternalSourceModuleProps() : ModuleProps() 
	{
		quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	}
};

class ExternalSourceModule : public Module
{
public:
	ExternalSourceModule(ExternalSourceModuleProps _props= ExternalSourceModuleProps()) : Module(SOURCE, "ExternalSourceModule", _props)
	{

	}

	virtual ~ExternalSourceModule() {}

	bool init()
	{
		if (!Module::init())
		{
			return false;
		}

		metadata = getFirstOutputMetadata();
		pinId = getOutputPinIdByType(metadata->getFrameType());

		return true;
	}

	// used in ut
	frame_sp makeFrame(size_t size, framemetadata_sp& metadata)
	{
		return Module::makeFrame(size, metadata);
	}

	// used in ut
	bool send(frame_container& frames)
	{
		return Module::send(frames);
	}

	// used in ut
	void sendEOS()
	{
		return Module::sendEOS();
	}

	pair<bool, uint64_t> produceExternalFrame(ApraData* data)
	{		
		frame_sp frame = frame_sp(new ExternalFrame(data));
		frame->setMetadata(metadata);
				
		frame_container frames;
		frames.insert(make_pair(pinId, frame));

		auto ret = Module::send(frames);		
		auto out = make_pair(ret, frame->fIndex);
		
		return out;
	}

	bool copyFrame(void* pBuffer, size_t size)
	{
		auto metadata = getFirstOutputMetadata();
		auto frame = Module::makeFrame(size, metadata);

		memcpy(frame->data(), pBuffer, size);

		frame_container frames;
		frames.insert(make_pair(getInputPinIdByType(metadata->getFrameType()), frame));

		return Module::send(frames);
	}

	bool stop()
	{
		if (isRunning())
		{
			return Module::stop();
		}
		else
		{
			sendEoPFrame();
			return true;
		}
	}

protected:
	bool validateOutputPins()
	{
		return true;
	}

private:
	framemetadata_sp metadata;
	std::string pinId;
};
