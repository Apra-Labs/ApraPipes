#pragma once
#include "Module.h"
#include "declarative/PropertyMacros.h"

class ThumbnailListGeneratorProps : public ModuleProps
{
public:
    // Default constructor for declarative pipeline
    ThumbnailListGeneratorProps() : thumbnailWidth(128), thumbnailHeight(128), fileToStore("")
    {
    }

    ThumbnailListGeneratorProps(int _thumbnailWidth, int _thumbnailHeight, std::string _fileToStore) : ModuleProps()
    {
        thumbnailWidth = _thumbnailWidth;
        thumbnailHeight = _thumbnailHeight;
        fileToStore = _fileToStore;
    }

    int thumbnailWidth;
    int thumbnailHeight;
    std::string fileToStore;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(fileToStore);
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
        apra::applyProp(props.thumbnailWidth, "thumbnailWidth", values, false, missingRequired);
        apra::applyProp(props.thumbnailHeight, "thumbnailHeight", values, false, missingRequired);
        apra::applyProp(props.fileToStore, "fileToStore", values, true, missingRequired);
    }

    apra::ScalarPropertyValue getProperty(const std::string& propName) const {
        if (propName == "thumbnailWidth") return static_cast<int64_t>(thumbnailWidth);
        if (propName == "thumbnailHeight") return static_cast<int64_t>(thumbnailHeight);
        if (propName == "fileToStore") return fileToStore;
        throw std::runtime_error("Unknown property: " + propName);
    }

    bool setProperty(const std::string& propName, const apra::ScalarPropertyValue& value) {
        return false;  // All properties are static
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
        ar &thumbnailWidth;
        ar &thumbnailHeight;
        ar &fileToStore;
    }
}; 
class ThumbnailListGenerator : public Module
{

public:
    ThumbnailListGenerator(ThumbnailListGeneratorProps _props);
    virtual ~ThumbnailListGenerator();
    bool init();
    bool term();
    void setProps(ThumbnailListGeneratorProps &props);
    ThumbnailListGeneratorProps getProps();

protected:
    bool process(frame_container &frames);
    bool validateInputPins();
    // bool processSOS(frame_sp &frame);
    // bool shouldTriggerSOS();
    bool handlePropsChange(frame_sp &frame);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};
