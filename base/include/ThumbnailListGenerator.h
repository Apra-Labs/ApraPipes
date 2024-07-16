#pragma once
#include "Module.h"

class ThumbnailListGeneratorProps : public ModuleProps
{
public:
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
