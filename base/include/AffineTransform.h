#pragma once

#include "Module.h"
#include <map>
#include <variant>
#include <string>
#ifdef APRA_HAS_CUDA_HEADERS
#include "CudaCommon.h"
#endif

class DetailMemoryAbstract;
class DeatilCUDA;
class DetailDMA;
class DetailHost;
class DetailGPU;

class AffineTransformProps : public ModuleProps
{
public:
	enum Interpolation {
		NN = 0,     
		LINEAR, 
		CUBIC,
		UNDEFINED,
		CUBIC2P_BSPLINE,
		CUBIC2P_CATMULLROM,
		CUBIC2P_B05C03,
		SUPER,
		LANCZOS,
		LANCZOS3_ADVANCED
	};

	enum TransformType
	{
		USING_OPENCV = 0,
		USING_NPPI
	};

	// Default constructor for declarative pipeline support
	AffineTransformProps()
	{
		type = TransformType::USING_OPENCV;
		angle = 0.0;
		x = 0;
		y = 0;
		scale = 1.0;
	}

	AffineTransformProps(TransformType _type, double _angle, int _x = 0, int _y = 0, double _scale = 1.0)
	{
		if (_type != TransformType::USING_OPENCV)
		{
			throw AIPException(AIP_FATAL, "This constructor only supports Opencv");
		}
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		type = _type;
	}

	// Declarative pipeline property binding
	void applyProperties(const std::map<std::string, std::variant<int64_t, double, bool, std::string>>& props)
	{
		for (const auto& [key, value] : props) {
			if (key == "angle") {
				if (std::holds_alternative<double>(value)) {
					angle = std::get<double>(value);
				} else if (std::holds_alternative<int64_t>(value)) {
					angle = static_cast<double>(std::get<int64_t>(value));
				}
			} else if (key == "x") {
				if (std::holds_alternative<int64_t>(value)) {
					x = static_cast<int>(std::get<int64_t>(value));
				}
			} else if (key == "y") {
				if (std::holds_alternative<int64_t>(value)) {
					y = static_cast<int>(std::get<int64_t>(value));
				}
			} else if (key == "scale") {
				if (std::holds_alternative<double>(value)) {
					scale = std::get<double>(value);
				} else if (std::holds_alternative<int64_t>(value)) {
					scale = static_cast<double>(std::get<int64_t>(value));
				}
			} else if (key == "transformType") {
				if (std::holds_alternative<std::string>(value)) {
					const auto& strVal = std::get<std::string>(value);
					if (strVal == "USING_OPENCV") {
						type = TransformType::USING_OPENCV;
					}
					// USING_NPPI not supported via TOML (requires cudastream_sp)
				}
			}
		}
	}

	std::variant<int64_t, double, bool, std::string> getProperty(const std::string& name) const
	{
		if (name == "angle") return angle;
		if (name == "x") return static_cast<int64_t>(x);
		if (name == "y") return static_cast<int64_t>(y);
		if (name == "scale") return scale;
		if (name == "transformType") return type == TransformType::USING_OPENCV ? std::string("USING_OPENCV") : std::string("USING_NPPI");
		return int64_t(0);
	}
#ifdef APRA_HAS_CUDA_HEADERS
	AffineTransformProps(TransformType _type, Interpolation _interpolation, cudastream_sp &_stream, double _angle, int _x=0, int _y=0, double _scale = 1.0)
	{
		if (_type != TransformType::USING_NPPI)
		{
			throw AIPException(AIP_FATAL, "This constructor only supports using NPPI");
		}
		stream = _stream;
		angle = _angle;
		x = _x;
		y = _y;
		scale = _scale;
		interpolation = _interpolation;
		type = _type;
	}
#endif

	int x = 0;
	int y = 0;
	double scale = 1.0;
	double angle;
#ifdef APRA_HAS_CUDA_HEADERS
	cudastream_sp stream;
#endif
	Interpolation interpolation = AffineTransformProps::NN;
	TransformType type;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(double) + sizeof(float) + sizeof(interpolation) + sizeof(type);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &angle &x &y &scale ;
		ar& interpolation;
		ar& type;
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
	AffineTransformProps mProp;
	bool handlePropsChange(frame_sp& frame);
	boost::shared_ptr<DetailMemoryAbstract> mDetail;
};