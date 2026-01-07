#pragma once

#include <opencv2/face.hpp>
#include "Module.h"
#include "declarative/PropertyMacros.h"

class Detail;
class DetailSSD;
class DetailHCASCADE;

class FacialLandmarkCVProps : public ModuleProps
{
public:
	enum FaceDetectionModelType
	{
		SSD,
		HAAR_CASCADE
	};

	// Default constructor for declarative pipeline
	FacialLandmarkCVProps() : type(SSD) {}

	FacialLandmarkCVProps(FaceDetectionModelType _type) : type(_type) {}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _Face_Detection_Configuration, const std::string _Face_Detection_Weights, const std::string _landmarksDetectionModel, cv::Ptr<cv::face::Facemark> _facemark)
		: type(_type), Face_Detection_Configuration(_Face_Detection_Configuration), Face_Detection_Weights(_Face_Detection_Weights), landmarksDetectionModel(_landmarksDetectionModel),facemark(_facemark)
	{
		if (_type != FaceDetectionModelType::SSD)
		{
			 throw AIPException(AIP_FATAL, "This constructor only supports SSD");
		}
	}

	FacialLandmarkCVProps(FaceDetectionModelType _type, const std::string _faceDetectionModel,const std::string _landmarksDetectionModel, cv::Ptr<cv::face::Facemark> _facemark)
		: type(_type), landmarksDetectionModel(_landmarksDetectionModel), faceDetectionModel(_faceDetectionModel), facemark(_facemark)
	{
		if (_type != FaceDetectionModelType::HAAR_CASCADE)
		{
			throw AIPException(AIP_FATAL, "This constructor only supports HAAR_CASCADE ");
		}
	}

	FaceDetectionModelType type;
	std::string Face_Detection_Configuration = "./data/assets/deploy.prototxt";
	std::string Face_Detection_Weights = "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	std::string landmarksDetectionModel = "./data/assets/face_landmark_model.dat";
	std::string faceDetectionModel = "./data/assets/haarcascade.xml";
	cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkKazemi::create();

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(type);
	}

	// ============================================================
	// Property Binding for Declarative Pipeline
	// ============================================================
	template<typename PropsT>
	static void applyProperties(
		PropsT& props,
		const std::map<std::string, apra::ScalarPropertyValue>& values,
		std::vector<std::string>& missingRequired
	) {
		// Handle FaceDetectionModelType enum
		auto it = values.find("modelType");
		if (it != values.end()) {
			if (auto* strVal = std::get_if<std::string>(&it->second)) {
				if (*strVal == "SSD") props.type = SSD;
				else if (*strVal == "HAAR_CASCADE") props.type = HAAR_CASCADE;
			}
		}
		apra::applyProp(props.Face_Detection_Configuration, "faceDetectionConfig", values, false, missingRequired);
		apra::applyProp(props.Face_Detection_Weights, "faceDetectionWeights", values, false, missingRequired);
		apra::applyProp(props.landmarksDetectionModel, "landmarksModel", values, false, missingRequired);
		apra::applyProp(props.faceDetectionModel, "haarCascadeModel", values, false, missingRequired);
	}

	apra::ScalarPropertyValue getProperty(const std::string& propName) const {
		if (propName == "modelType") return type == SSD ? std::string("SSD") : std::string("HAAR_CASCADE");
		if (propName == "faceDetectionConfig") return Face_Detection_Configuration;
		if (propName == "faceDetectionWeights") return Face_Detection_Weights;
		if (propName == "landmarksModel") return landmarksDetectionModel;
		if (propName == "haarCascadeModel") return faceDetectionModel;
		throw std::runtime_error("Unknown property: " + propName);
	}

	bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
		// All properties are static (models loaded at init)
		return false;
	}

	static std::vector<std::string> dynamicPropertyNames() {
		return {};  // No dynamically changeable properties
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& type;
	}
};

class FacialLandmarkCV : public Module
{
 public:
	FacialLandmarkCV(FacialLandmarkCVProps props);
	virtual ~FacialLandmarkCV();
	bool init();
	bool term();
	void setProps(FacialLandmarkCVProps& props);
	FacialLandmarkCVProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	boost::shared_ptr<Detail> mDetail;
	FacialLandmarkCVProps mProp;
	bool handlePropsChange(frame_sp& frame);
	std::string mOutputPinId;
};