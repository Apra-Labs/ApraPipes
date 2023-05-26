#pragma once

#include "Module.h"
#include "CudaCommon.h"

class Detail;
class DeatilCUDA;
class DetailDMA;
class DetailHost;

class AffineTransformProps : public ModuleProps
{
public:
	enum Interpolation {

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

	AffineTransformProps( double _angle, double _mscale,int _x = 0, int _y = 0)
	{
		angle = _angle;
		x = _x;
		y = _y;
		mscale = _mscale;
	}

	AffineTransformProps(cudastream_sp& _stream, double _angle, int _x = 0, int _y = 0, float _scale = 1.0f)
	{
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
	}

	AffineTransformProps(Interpolation _eInterpolation, cudastream_sp &_stream, double _angle, int _x=0, int _y=0, float _scale=1.0f)
	{
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		eInterpolation = _eInterpolation;
	}

	int x = 0;// shiftX
	int y = 0; // shiftY
	double mscale;
	float scale = 1.0f;
	double angle;
	cudastream_sp stream;
	Interpolation eInterpolation = AffineTransformProps:: NN;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(double) + sizeof(float) + sizeof(eInterpolation);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &angle &x &y &scale ;
		ar& eInterpolation;
	}
};

class AffineTransform : public Module
{

public:
	AffineTransform(AffineTransformProps props);
	virtual ~AffineTransform();
	bool init();
	bool term();
	void setProps(AffineTransformProps &props);
	AffineTransformProps getProps();
	
protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	void setProps(AffineTransform);
	AffineTransformProps mProp;
	bool handlePropsChange(frame_sp &frame);
	boost::shared_ptr<Detail> mDetail;
};