#pragma once
#include "Module.h"
#include <map>
#include <vector>
#include "declarative/PropertyMacros.h"

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps() {}
    TestSignalGeneratorProps(int _width, int _height)
        : width(_width), height(_height) {}

    ~TestSignalGeneratorProps() {}

    int width = 0;
    int height = 0;

    // ============================================================
    // Property Binding for Declarative Pipeline
    // ============================================================
    template<typename PropsT>
    static void applyProperties(
        PropsT& props,
        const std::map<std::string, apra::ScalarPropertyValue>& values,
        std::vector<std::string>& missingRequired
    ) {
        apra::applyProp(props.width, "width", values, true, missingRequired);
        apra::applyProp(props.height, "height", values, true, missingRequired);
    }

    apra::ScalarPropertyValue getProperty(const std::string& propName) const {
        if (propName == "width") return static_cast<int64_t>(width);
        if (propName == "height") return static_cast<int64_t>(height);
        throw std::runtime_error("Unknown property: " + propName);
    }

    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
        throw std::runtime_error("Cannot modify static property '" + propName + "' after initialization");
    }

    static std::vector<std::string> dynamicPropertyNames() {
        return {};
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &width;
        ar &height;
    }
};

class TestSignalGenerator : public Module
{
public:
    TestSignalGenerator(TestSignalGeneratorProps _props);
    ~TestSignalGenerator();

    bool init();
    bool term();
    void setProps(TestSignalGeneratorProps &props);
    TestSignalGeneratorProps getProps();

protected:
    bool produce();
    bool validateOutputPins();
    void setMetadata(framemetadata_sp &metadata);
    bool handlePropsChange(frame_sp &frame);


private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    size_t outputFrameSize;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
};
