#pragma once

#include "Module.h"

class BayerToRGBAProps : public ModuleProps
{
public:
	BayerToRGBAProps() 
	{
	}

};

class BayerToRGBA : public Module
{

public:
	BayerToRGBA(BayerToRGBAProps _props);
	virtual ~BayerToRGBA();
	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); 
	std::string addOutputPin(framemetadata_sp &metadata);


private:		
	void setMetadata(framemetadata_sp& metadata);
	int mFrameType;
	BayerToRGBAProps props;
	class Detail;
	boost::shared_ptr<Detail> mDetail;			
	size_t mMaxStreamLength;
};