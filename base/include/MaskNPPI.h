#pragma once

#include "Module.h"
#include "CudaCommon.h"

class MaskNPPIProps : public ModuleProps
{
public:
	enum AVAILABLE_MASKS
	{
		NONE,
		CIRCLE,
		OCTAGONAL
	};

	MaskNPPIProps(int _centerX, int _centerY, int _radius, cudastream_sp &_stream) : centerX(_centerX), centerY(_centerY), radius(_radius)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	MaskNPPIProps(int _centerX, int _centerY, int _radius, AVAILABLE_MASKS _availableMask, cudastream_sp &_stream) : centerX(_centerX), centerY(_centerY), radius(_radius)
	{
		maskSelected = _availableMask;
		stream_sp = _stream;
		stream = _stream->getCudaStream();
	}

	AVAILABLE_MASKS maskSelected = MaskNPPIProps::NONE;
	cudaStream_t stream;
	cudastream_sp stream_sp;
	int centerX;
	int centerY;
	int radius;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(stream) + sizeof(maskSelected) + 2 * sizeof(centerX);
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
		ar & maskSelected;
		ar & centerX;
		ar & centerY;
		ar & radius;
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
