#pragma once
#include "Module.h"
#include <map>
#include <vector>
#include "declarative/PropertyMacros.h"

// Pattern types for test signal generation
enum class TestPatternType {
    GRADIENT = 0,      // Horizontal gradient bands (default)
    CHECKERBOARD = 1,  // Black/white checkerboard
    COLOR_BARS = 2,    // Vertical color bars
    GRID = 3           // Numbered grid cells
};

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps() {}
    TestSignalGeneratorProps(int _width, int _height, TestPatternType _pattern = TestPatternType::GRADIENT, int _maxFrames = 0)
        : width(_width), height(_height), pattern(_pattern), maxFrames(_maxFrames) {}

    ~TestSignalGeneratorProps() {}

    int width = 0;
    int height = 0;
    TestPatternType pattern = TestPatternType::GRADIENT;
    int maxFrames = 0;  // 0 = unlimited, >0 = stop after N frames

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
        apra::applyProp(props.maxFrames, "maxFrames", values, false, missingRequired);

        // Handle pattern property (optional, default GRADIENT)
        auto patternIt = values.find("pattern");
        if (patternIt != values.end()) {
            if (std::holds_alternative<std::string>(patternIt->second)) {
                std::string patternStr = std::get<std::string>(patternIt->second);
                if (patternStr == "GRADIENT") props.pattern = TestPatternType::GRADIENT;
                else if (patternStr == "CHECKERBOARD") props.pattern = TestPatternType::CHECKERBOARD;
                else if (patternStr == "COLOR_BARS") props.pattern = TestPatternType::COLOR_BARS;
                else if (patternStr == "GRID") props.pattern = TestPatternType::GRID;
            } else if (std::holds_alternative<int64_t>(patternIt->second)) {
                props.pattern = static_cast<TestPatternType>(std::get<int64_t>(patternIt->second));
            }
        }
    }

    apra::ScalarPropertyValue getProperty(const std::string& propName) const {
        if (propName == "width") return static_cast<int64_t>(width);
        if (propName == "height") return static_cast<int64_t>(height);
        if (propName == "maxFrames") return static_cast<int64_t>(maxFrames);
        if (propName == "pattern") {
            switch (pattern) {
                case TestPatternType::GRADIENT: return std::string("GRADIENT");
                case TestPatternType::CHECKERBOARD: return std::string("CHECKERBOARD");
                case TestPatternType::COLOR_BARS: return std::string("COLOR_BARS");
                case TestPatternType::GRID: return std::string("GRID");
            }
        }
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
        ar &pattern;
        ar &maxFrames;
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
