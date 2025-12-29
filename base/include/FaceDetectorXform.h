#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>
#include <array>
#include "declarative/Metadata.h"

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
    // ============================================================
    // Declarative Pipeline Metadata
    // ============================================================
    struct Metadata {
        static constexpr std::string_view name = "FaceDetectorXform";
        static constexpr apra::ModuleCategory category = apra::ModuleCategory::Analytics;
        static constexpr std::string_view version = "1.0.0";
        static constexpr std::string_view description =
            "Detects faces in image frames using deep learning models. "
            "Outputs face bounding boxes and confidence scores.";

        static constexpr std::array<std::string_view, 4> tags = {
            "analytics", "face", "detection", "transform"
        };

        static constexpr std::array<apra::PinDef, 1> inputs = {
            apra::PinDef::create("input", "RawImagePlanar", true, "Image frames to process")
        };

        static constexpr std::array<apra::PinDef, 1> outputs = {
            apra::PinDef::create("output", "Frame", true, "Frames with face detection metadata")
        };

        static constexpr std::array<apra::PropDef, 2> properties = {
            apra::PropDef::Float("scaleFactor", 1.0, 0.1, 10.0,
                "Scale factor for input image preprocessing"),
            apra::PropDef::Float("confidenceThreshold", 0.5, 0.0, 1.0,
                "Minimum confidence threshold for face detection")
        };
    };

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