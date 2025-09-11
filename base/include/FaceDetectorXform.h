#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>

class FaceDetectorXformProps : public ModuleProps
{
public:
    FaceDetectorXformProps(double _scaleFactor = 1.0, float _confidenceThreshold = 0.5, std::string _Face_Detection_Configuration= "./data/assets/deploy.prototxt", std::string _Face_Detection_Weights= "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        : scaleFactor(_scaleFactor), confidenceThreshold(_confidenceThreshold), FACE_DETECTION_CONFIGURATION(_Face_Detection_Configuration), FACE_DETECTION_WEIGHTS(_Face_Detection_Weights)
    {
    }
    double scaleFactor;
    float confidenceThreshold;
    std::string FACE_DETECTION_CONFIGURATION;
	std::string FACE_DETECTION_WEIGHTS;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(scaleFactor) + sizeof(confidenceThreshold);
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &scaleFactor &confidenceThreshold;
    }
};

class FaceDetectorXform : public Module
{
public:
    FaceDetectorXform(FaceDetectorXformProps props);
    virtual ~FaceDetectorXform() {}

    virtual bool init();
    virtual bool term();

    void setProps(FaceDetectorXformProps &props);
    FaceDetectorXformProps getProps();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void setMetadata(framemetadata_sp &metadata);
    void addInputPin(framemetadata_sp &metadata, string &pinId);
    bool shouldTriggerSOS();
    bool handlePropsChange(frame_sp &frame);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};