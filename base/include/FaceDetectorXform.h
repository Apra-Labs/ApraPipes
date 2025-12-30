#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>
#include <array>
#include <map>
#include <vector>
#include <algorithm>
#include "declarative/Metadata.h"
#include "declarative/PropertyMacros.h"

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

    // ============================================================
    // Property Binding for Declarative Pipeline
    // Both properties are Dynamic (can change at runtime)
    // ============================================================
    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        // scaleFactor is optional (has default)
        apra::applyProp(props.scaleFactor, "scaleFactor", values, false, missingRequired);
        // confidenceThreshold is optional (has default)
        apra::applyProp(props.confidenceThreshold, "confidenceThreshold", values, false, missingRequired);
    }

    // Runtime property getter
    apra::ScalarPropertyValue getProperty(const std::string& propName) const {
        if (propName == "scaleFactor") return scaleFactor;
        if (propName == "confidenceThreshold") return static_cast<double>(confidenceThreshold);
        throw std::runtime_error("Unknown property: " + propName);
    }

    // Runtime property setter (both properties are Dynamic)
    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
        if (propName == "scaleFactor") {
            return apra::applyFromVariant(scaleFactor, value);
        }
        if (propName == "confidenceThreshold") {
            return apra::applyFromVariant(confidenceThreshold, value);
        }
        throw std::runtime_error("Unknown property: " + propName);
    }

    // Both properties are dynamic
    static std::vector<std::string> dynamicPropertyNames() {
        return {"scaleFactor", "confidenceThreshold"};
    }

    // Check if a property is dynamic
    static bool isPropertyDynamic(const std::string& propName) {
        auto names = dynamicPropertyNames();
        return std::find(names.begin(), names.end(), propName) != names.end();
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
            apra::PropDef::Floating("scaleFactor", 1.0, 0.1, 10.0,
                "Scale factor for input image preprocessing"),
            apra::PropDef::Floating("confidenceThreshold", 0.5, 0.0, 1.0,
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