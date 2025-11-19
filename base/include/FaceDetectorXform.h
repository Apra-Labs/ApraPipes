#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>

class FaceDetectorXformProps : public ModuleProps
{
public:
    FaceDetectorXformProps(double _scaleFactor = 1.0, float _confidenceThreshold = 0.5) : scaleFactor(_scaleFactor), confidenceThreshold(_confidenceThreshold)
    {
    }
    double scaleFactor;
    float confidenceThreshold;

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
    std::shared_ptr<Detail> mDetail;
};