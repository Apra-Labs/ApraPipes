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
	bool init() { return Module::init(); }
	bool term() { return Module::term(); }
protected:
	bool process(frame_container& frames) { return true; }	
	bool validateInputPins() { return true; }
private:	
};