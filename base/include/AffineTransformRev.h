#pragma once

#include "Module.h"
#include "CudaCommon.h"

class AffineTransformRevProps : public ModuleProps
{
public:
	enum Interpolation
	{
		NN,
		LINEAR,
		CUBIC,
		UNDEFINED,
		CUBIC2P_BSPLINE,
		CUBIC2P_CATMULLROM,
		CUBIC2P_B05C03,
		SUPER,
		LANCZOS,
		LANCZOS3_ADVANCED,
	};
	AffineTransformRevProps(cudastream_sp &_stream, double _angle, int _x = 0, int _y = 0, float _scale = 1.0f)
	{
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		step = 0;
	}

	AffineTransformRevProps(Interpolation _eInterpolation, cudastream_sp &_stream, double _angle, int _x = 0, int _y = 0, float _scale = 1.0f)
	{
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		eInterpolation = _eInterpolation;
		step = 0; 
	}

	AffineTransformRevProps(Interpolation _eInterpolation, cudastream_sp &_stream, double _angle, int _step, int _x = 0, int _y = 0, float _scale = 1.0)
	{
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		eInterpolation = _eInterpolation;
		step = _step;
	}

	int x = 0;
	int y = 0;
	float scale = 1.0f;
	double angle;
	cudastream_sp stream;
	Interpolation eInterpolation = AffineTransformRevProps::NN;
	int step;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 3 + sizeof(double) + sizeof(float) + sizeof(Interpolation);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &angle; 
		ar &x;
		ar &y; 
		ar &scale;
		ar &eInterpolation;
		ar &step;
	}
};

class AffineTransformRev : public Module
{

public:
	AffineTransformRev(AffineTransformRevProps props);
	virtual ~AffineTransformRev();
	bool init();
	bool term();
	void setProps(AffineTransformRevProps &props);
	AffineTransformRevProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	void setProps(AffineTransformRev);
	bool handlePropsChange(frame_sp &frame);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};