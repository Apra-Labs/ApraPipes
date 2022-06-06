#pragma once

#include "Module.h"

class ArgusStatsProps : public ModuleProps
{
public:
	ArgusStatsProps(uint8_t _saturatedPixel) 
	{
		saturatedPixel = _saturatedPixel;
	}
	uint8_t saturatedPixel;
};

class ArgusStats : public Module
{

public:
	ArgusStats(ArgusStatsProps _props);
	virtual ~ArgusStats();
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
	ArgusStatsProps props;
	class Detail;
	boost::shared_ptr<Detail> mDetail;			
	size_t mMaxStreamLength;
};