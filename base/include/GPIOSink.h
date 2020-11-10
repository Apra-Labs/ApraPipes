#pragma once
#include "Module.h"
#include <chrono>

class GPIODriver;

class GPIOSinkProps : public ModuleProps
{
public:
	GPIOSinkProps(uint32_t _gpioNo, float _highTime) : ModuleProps()
	{
		gpioNo = _gpioNo;
		highTime = _highTime;
	}

	float highTime; //ms
	uint32_t gpioNo;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(highTime) + sizeof(gpioNo);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &highTime;
		ar &gpioNo;
	}
};

class GPIOSink : public Module
{
public:
	GPIOSink(GPIOSinkProps _props);
	virtual ~GPIOSink();
	bool init();
	bool term();

	void setProps(GPIOSinkProps &props);
	GPIOSinkProps getProps();

protected:
	bool process(frame_container &frames);
	bool validateInputPins();
	bool handlePropsChange(frame_sp &frame);

private:
	bool initDriver();

	std::chrono::milliseconds mSleepTime;
	boost::shared_ptr<GPIODriver> mDriver;
	GPIOSinkProps mProps;
};
