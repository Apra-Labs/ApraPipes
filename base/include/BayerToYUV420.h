#pragma once
 
#include "Module.h"
 
class BayerToYUV420Props : public ModuleProps
{
public:
    BayerToYUV420Props()
    {
    }
 
};
 
class BayerToYUV420 : public Module
{
 
public:
    BayerToYUV420(BayerToYUV420Props _props);
    virtual ~BayerToYUV420();
    bool init();
    bool term();
protected:
    bool process(frame_container& frames);
    bool processSOS(frame_sp& frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp& metadata, string& pinId);
    std::string addOutputPin(framemetadata_sp &metadata);
 
 
private:        
    void setMetadata(framemetadata_sp& metadata);
    int mFrameType;
    BayerToYUV420Props props;
    class Detail;
    boost::shared_ptr<Detail> mDetail;          
    size_t mMaxStreamLength;
};
