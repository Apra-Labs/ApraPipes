#pragma once

#include "Module.h"

class StatSinkProps : public ModuleProps
{
public:
	StatSinkProps() : ModuleProps() {}
};

class StatSink : public Module {
public:	
	StatSink(StatSinkProps _props = StatSinkProps()): Module(SINK, "StatSink", _props) {}
	virtual ~StatSink() {}
	bool init() override { return Module::init(); }
	bool term() override { return Module::term(); }
protected:
	bool process(frame_container& frames) override { return true; }
	bool validateInputPins() override { return true; }
private:	
};