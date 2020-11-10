#include "GPIOSink.h"
#include "FrameMetadata.h"
#include "FileSequenceDriver.h"
#include "Frame.h"
#include "GPIODriver.h"
#include <chrono>

GPIOSink::GPIOSink(GPIOSinkProps _props)
	: Module(SINK, "GPIOSink", _props), mProps(_props)
{
}

GPIOSink::~GPIOSink() {}

bool GPIOSink::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		return false;
	}

	// any input type is fine

	return true;
}

bool GPIOSink::initDriver()
{
	if (!mProps.gpioNo)
	{
		return true;
	}

	if (mProps.highTime < 1)
	{
		mProps.highTime = 1;
	}	
	mSleepTime = std::chrono::milliseconds(static_cast<int>(mProps.highTime));

	mDriver.reset(new GPIODriver(mProps.gpioNo));
	if (!mDriver->Init(false))
	{
		return false;
	}

	// making the pin low by default
	return mDriver->Write(false);
}

bool GPIOSink::init()
{
	if (!Module::init())
	{
		return false;
	}

	return initDriver();
}

bool GPIOSink::term()
{
	mDriver.reset();

	return Module::term();
}

bool GPIOSink::process(frame_container &frames)
{
	if (!mProps.gpioNo)
	{
		return true;
	}

	mDriver->Write(true);
	std::this_thread::sleep_for(mSleepTime);	
	mDriver->Write(false);

	return true;
}

void GPIOSink::setProps(GPIOSinkProps &props)
{
	Module::setProps(props, PropsChangeMetadata::ModuleName::GPIOSink);
}

GPIOSinkProps GPIOSink::getProps()
{
	fillProps(mProps);
	return mProps;
}

bool GPIOSink::handlePropsChange(frame_sp &frame)
{
	bool ret = Module::handlePropsChange(frame, mProps);

	return initDriver();
}