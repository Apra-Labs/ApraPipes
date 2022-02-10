#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>

class FaceDetectorProps : public ModuleProps
{
public:
    FaceDetectorProps(std::string _binPath = "../../data/version-RFB/RFB-320.bin", std::string _paramPath = "../../data/version-RFB/RFB-320.param") : binPath(_binPath), paramPath(_paramPath)
    {
    }
    std::string binPath;
    std::string paramPath;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(binPath) + sizeof(paramPath) + binPath.length() + paramPath.length();
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &binPath &paramPath;
    }
};

class FaceDetector : public Module
{
public:
    FaceDetector(FaceDetectorProps props);
    virtual ~FaceDetector() {}

    virtual bool init();
    virtual bool term();

    void setProps(FaceDetectorProps &props);
    FaceDetectorProps getProps();

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