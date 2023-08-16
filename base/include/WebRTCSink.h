#pragma once 
#include "Module.h"
#include "FrameContainerQueue.h"
#include "BaresipWebRTC.h"
#include<stdlib.h>  
#include<sys/shm.h> 

//This module gives frames from the input to baresip vidpipe source module.

class WebRTCSinkProps : public ModuleProps
{
public:
	WebRTCSinkProps()
	{

	}
};

class WebRTCSink : public Module
{
public:
	WebRTCSink(WebRTCSinkProps _props);
	~WebRTCSink();
	bool init();
	bool term();
	void setProps(WebRTCSinkProps& props);
	WebRTCSinkProps getProps(); 

protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();

private:
	void setMetadata(framemetadata_sp& metadata);
	boost::shared_ptr<BaresipWebRTC> adapter = boost::shared_ptr<BaresipWebRTC>(new BaresipWebRTC(BaresipWebRTCProps()));
	class Detail;
	boost::shared_ptr<Detail> mDetail;

};