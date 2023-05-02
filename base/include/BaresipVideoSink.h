#pragma once 
#include "Module.h"
#include "FrameContainerQueue.h"
#include "BaresipVideoAdapter.h"
#include<stdlib.h>  
#include<sys/shm.h> 

//This module gives frames from the input to baresip vidpipe source module.

class BaresipVideoSinkProps : public ModuleProps
{
public:
	BaresipVideoSinkProps()
	{

	}
};

class BaresipVideoSink : public Module
{
public:
	BaresipVideoSink(BaresipVideoSinkProps _props);
	~BaresipVideoSink();
	bool init();
	bool term();
	void setProps(BaresipVideoSinkProps& props);
	BaresipVideoSinkProps getProps(); 

protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();

private:
	void setMetadata(framemetadata_sp& metadata);
	boost::shared_ptr<BaresipVideoAdapter> adapter = boost::shared_ptr<BaresipVideoAdapter>(new BaresipVideoAdapter(BaresipVideoAdapterProps()));
	class Detail;
	boost::shared_ptr<Detail> mDetail;

};