#pragma once

#include "Module.h"
#include "CudaCommon.h"

class OverlayNPPIProps : public ModuleProps
{
public:
	OverlayNPPIProps(cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		offsetX = 0;
		offsetY = 0;
		globalAlpha = -1;
	}

	int offsetX;
	int offsetY;
	int globalAlpha;
	cudaStream_t stream;
	cudastream_sp stream_sp;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(offsetX) + sizeof(offsetY) + sizeof(globalAlpha) + sizeof(stream);
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
		ar & offsetX;
		ar & offsetY;
		ar & globalAlpha;
	}
};

class OverlayNPPI : public Module
{

public:
	OverlayNPPI(OverlayNPPIProps _props);
	virtual ~OverlayNPPI();
	bool init();
	bool term();

	void setProps(OverlayNPPIProps& props);
	OverlayNPPIProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputOutputPins();
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
	OverlayNPPIProps props;		
};
