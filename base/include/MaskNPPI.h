#pragma once

#include "Module.h"
#include "CudaCommon.h"

class MaskNPPIProps : public ModuleProps
{
public:
	MaskNPPIProps(cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}
	

	cudaStream_t stream;
	cudastream_sp stream_sp;


	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(stream);
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
	}
};

class MaskNPPI : public Module
{

public:
	MaskNPPI(MaskNPPIProps _props);
	virtual ~MaskNPPI();
	bool init();
	bool term();

	void setProps(MaskNPPIProps& props);
	MaskNPPIProps getProps();

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
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	MaskNPPIProps props;		
};
