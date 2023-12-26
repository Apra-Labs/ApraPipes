#pragma once

#include "Module.h"
#include "CudaCommon.h"

class RotationIndicatorProps : public ModuleProps
{
public:
    enum AVAILABLE_MASKS
    {
        NONE,
        CIRCLE,
        OCTAGONAL
    };

    RotationIndicatorProps(double _rotationAngle, bool _enableMask, cudastream_sp &_stream) : rotationAngle(_rotationAngle), enableMask(_enableMask)
    {
        stream_sp = _stream;
        stream = _stream->getCudaStream();
    }

    AVAILABLE_MASKS maskSelected = RotationIndicatorProps::NONE;
    cudaStream_t stream;
    cudastream_sp stream_sp;
    double rotationAngle;
    bool enableMask = false;

    size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize() + sizeof(stream) + sizeof(maskSelected) + sizeof(rotationAngle) + sizeof(enableMask);
    }

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar & maskSelected;
        ar & rotationAngle;
        ar & enableMask;
    }
};

class RotationIndicatorKernel : public Module
{

public:
    RotationIndicatorKernel(RotationIndicatorProps _props);
    virtual ~RotationIndicatorKernel();
    bool init();
    bool term();
    void setProps(RotationIndicatorProps &props);
    RotationIndicatorProps getProps();

protected:
    bool process(frame_container &frames);
    bool processSOS(frame_sp &frame);
    bool validateInputPins();
    bool validateOutputPins();
    void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
    bool shouldTriggerSOS();
    bool processEOS(string &pinId);
    bool handlePropsChange(frame_sp &frame);

private:
    void setMetadata(framemetadata_sp &metadata);

    class Detail;
    boost::shared_ptr<Detail> mDetail;

    int mFrameType;
    int mInputFrameType;
    int mOutputFrameType;
    size_t mFrameLength;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    RotationIndicatorProps props;
};
