#pragma once

#include "Module.h"
#include "CudaCommon.h"

class EffectsNPPIProps : public ModuleProps
{
public:
	EffectsNPPIProps(cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		contrast = 1;
		brightness = 0;
		hue = 0;
		saturation = 1;
	}
	
	double hue;            // 0/255 no change [0 255] 
    double saturation;     // 1 no change
    double contrast;       // 1 no change 
    int brightness;     // 0 no change [-100 - 100]

	cudaStream_t stream;
	cudastream_sp stream_sp;


	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(contrast) + sizeof(brightness) + sizeof(hue) + sizeof(saturation) + sizeof(stream);
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
		ar & contrast;
		ar & brightness;
		ar & hue;
		ar & saturation;
	}
};

class EffectsNPPI : public Module
{

public:
	EffectsNPPI(EffectsNPPIProps _props);
	virtual ~EffectsNPPI();
	bool init();
	bool term();

	void setProps(EffectsNPPIProps& props);
	EffectsNPPIProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);

	class Detail;
	boost::shared_ptr<Detail> mDetail;

	int mFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	EffectsNPPIProps props;		
};
