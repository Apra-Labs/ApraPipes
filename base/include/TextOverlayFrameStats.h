#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include <deque>

class TextOverlayFrameStatsProps : public ModuleProps
{
public:
    TextOverlayFrameStatsProps()
    {
    }
};

class TextOverlayFrameStats : public Module
{

public:
    TextOverlayFrameStats(TextOverlayFrameStatsProps _props);
    virtual ~TextOverlayFrameStats();
    bool init();
    bool term();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
    bool shouldTriggerSOS();
    bool processEOS(string &pinId);

private:
    void setMetadata(framemetadata_sp &metadata);
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    size_t mFrameChecker;

    TextOverlayFrameStatsProps props;
};