#pragma once
#include "FrameMetadata.h"
#include "Module.h"

class PerspectiveTransformProps : public ModuleProps
{
public:
    PerspectiveTransformProps(const std::vector<cv::Point2f> &_srcPoints, const std::vector<cv::Point2f> &_dstPoints)
    {
        srcPoints = _srcPoints;
        dstPoints = _dstPoints;
    }
    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
};

class PerspectiveTransform : public Module
{
public:
    PerspectiveTransform(PerspectiveTransformProps _props);
    virtual ~PerspectiveTransform();
    bool init();
    bool term();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp &metadata, string &pinId);
    std::string addOutputPin(framemetadata_sp &metadata);

private:
    int mFrameType;
    PerspectiveTransformProps mProps;
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};