#pragma once

#include "Module.h"

class BayerToGrayProps : public ModuleProps
{
public:
	BayerToGrayProps() 
	{
	}

};

class BayerToGray : public Module
{

public:
	BayerToGray(BayerToGrayProps _props);
	virtual ~BayerToGray();
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
	BayerToGrayProps props;
	class Detail;
	boost::shared_ptr<Detail> mDetail;			
	size_t mMaxStreamLength;
};