#pragma once
#include "FrameMetadata.h"
#include "Module.h"
#include "Logger.h"

class PerspectiveTransformProps : public ModuleProps
{
public:
    enum Mode
    {
        BASIC,
        DYNAMIC
    };

    // Default constructor for serialization
    PerspectiveTransformProps() : mode(BASIC) {}

    // Constructor for BASIC mode
    PerspectiveTransformProps(const std::vector<cv::Point2f> &_srcPoints, const std::vector<cv::Point2f> &_dstPoints)
        : mode(BASIC)
    {
        srcPoints = _srcPoints;
        dstPoints = _dstPoints;
    }

    // Constructor for DYNAMIC mode
    PerspectiveTransformProps(Mode _mode) : mode(_mode)
    {
        if (_mode != DYNAMIC)
        {
            throw AIPException(AIP_FATAL, "This constructor only supports DYNAMIC mode");
        }
    }

    Mode mode;
    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(Mode) + 
               sizeof(size_t) + srcPoints.size() * sizeof(cv::Point2f) +
               sizeof(size_t) + dstPoints.size() * sizeof(cv::Point2f);
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &mode;
        
        size_t srcSize = srcPoints.size();
        size_t dstSize = dstPoints.size();
        ar &srcSize &dstSize;
        
        for (size_t i = 0; i < srcSize; ++i) {
            ar &srcPoints[i].x &srcPoints[i].y;
        }
        for (size_t i = 0; i < dstSize; ++i) {
            ar &dstPoints[i].x &dstPoints[i].y;
        }
    }
    
    template <class Archive>
    void load(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &mode;
        
        size_t srcSize, dstSize;
        ar &srcSize &dstSize;
        
        srcPoints.resize(srcSize);
        dstPoints.resize(dstSize);
        
        for (size_t i = 0; i < srcSize; ++i) {
            ar &srcPoints[i].x &srcPoints[i].y;
        }
        for (size_t i = 0; i < dstSize; ++i) {
            ar &dstPoints[i].x &dstPoints[i].y;
        }
    }
    
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

class PerspectiveTransform : public Module
{
public:
    PerspectiveTransform(PerspectiveTransformProps _props);
    virtual ~PerspectiveTransform();
    bool init();
    bool term();
    void setProps(PerspectiveTransformProps &props);
    PerspectiveTransformProps getProps();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp &metadata, string &pinId);
    std::string addOutputPin(framemetadata_sp &metadata);
    bool handlePropsChange(frame_sp &frame);

private:
    int mFrameType;
    PerspectiveTransformProps mProps;
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};