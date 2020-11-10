#pragma once
#include <string>
#include <unordered_map>
#include "Module.h"

class GPIODriver;

class GPIOSourceProps : public ModuleProps
{
public:
	GPIOSourceProps(uint32_t _gpioNo) : ModuleProps(0)
	{
		gpioNo = _gpioNo;
		highTime = 0;
	}

	GPIOSourceProps(uint32_t _gpioNo, uint32_t _highTime) : ModuleProps(0)
	{
		gpioNo = _gpioNo;
		highTime = _highTime;
	}

	uint32_t highTime; //ms
	uint32_t gpioNo;
};

class GPIOSource : public Module
{
public:
	GPIOSource(GPIOSourceProps _props);
	virtual ~GPIOSource();
	bool init();
	bool term();

protected:
	bool produce();
	bool validateOutputPins();
	void notifyPlay(bool play);

private:
	bool sendValue(int value);

	class Detail;
	boost::shared_ptr<Detail> mDetail;
	boost::shared_ptr<GPIODriver> mDriver;
	GPIOSourceProps mProps;

	size_t mOutputSize;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};