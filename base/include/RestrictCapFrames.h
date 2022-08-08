#pragma once

#include "Module.h"
#include "CudaCommon.h"
#include <chrono>

#include <deque>

class RestrictCapFramesProps : public ModuleProps
{
public:
    RestrictCapFramesProps(int _noOfframesToCapture)
    {
        noOfframesToCapture = _noOfframesToCapture;
    }
    int noOfframesToCapture;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(noOfframesToCapture);
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &noOfframesToCapture;
    }
};

class RestrictCapFrames : public Module
{

public:
    RestrictCapFrames(RestrictCapFramesProps _props);
    virtual ~RestrictCapFrames();
    bool init();
    bool term();
    bool resetFrameCapture();
    void setProps(RestrictCapFramesProps &props);
    RestrictCapFramesProps getProps();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
    bool handleCommand(Command::CommandType type, frame_sp &frame);
    bool handlePropsChange(frame_sp &frame);

private:
    void setMetadata(framemetadata_sp &metadata);
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    class RestrictCapFramesResetCommand;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
};