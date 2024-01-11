#pragma once

#include "Module.h"
#include "CudaCommon.h"

class SquareMaskNPPIProps : public ModuleProps
{
public:

	SquareMaskNPPIProps(int _maskSize, cudastream_sp &_stream) : maskSize(_maskSize)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	cudaStream_t stream;
	cudastream_sp stream_sp;
    int maskSize;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(stream) + sizeof(maskSize);
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
        ar & maskSize;
	}
};

class SquareMaskNPPI : public Module
{

public:
	SquareMaskNPPI(SquareMaskNPPIProps _props);
	virtual ~SquareMaskNPPI();
	bool init();
	bool term();

	void setProps(SquareMaskNPPIProps& props);
	SquareMaskNPPIProps getProps();

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
	SquareMaskNPPIProps props;		
};
