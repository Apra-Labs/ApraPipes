#pragma once
#include "Module.h"

using CallbackFunction = std::function<void()>;
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
    void registerCallback(const CallbackFunction &_callback)
	{
		m_callbackFunction = _callback;
	}
    bool init();
    bool term();
    void setProps(ThumbnailListGeneratorProps &props);
    ThumbnailListGeneratorProps getProps();
    std::vector<unsigned char>getFrameBuffer();
    void setMetadata(std::string data);

protected:
    bool process(frame_container &frames);
    bool validateInputPins();
    // bool processSOS(frame_sp &frame);
    // bool shouldTriggerSOS();
    bool handlePropsChange(frame_sp &frame);
    void decompressFrame();

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    std::vector<unsigned char> m_frameBuffer;
    std::string m_customMetadata;
    CallbackFunction m_callbackFunction = NULL;
    std::string calculateMD5(const unsigned char *data, size_t length);
};
