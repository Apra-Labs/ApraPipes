#pragma once

#include "Module.h"

class DummyDMASourceProps : public ModuleProps
{
public:
	DummyDMASourceProps(std::string _fileName, int _width, int _height) : ModuleProps()
	{
		fileName = _fileName;
		width = _width;
		height = _height;
	}
	std::string fileName;
	int width;
	int height;
};

class DummyDMASource : public Module
{
public:
	DummyDMASource(DummyDMASourceProps _props);
	virtual ~DummyDMASource();
	bool init();
	bool term();

protected:
	bool produce();
	bool validateOutputPins();

private:
	std::string mOutputPinId;
	DummyDMASourceProps mProps;
};
