#pragma once

#include "Module.h"
#include "CudaCommon.h"

// #include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class RandomDMAProps : public ModuleProps
{
public:
    RandomDMAProps(cudastream_sp &_stream)
    {
        stream_sp = _stream;
        stream = _stream->getCudaStream();
    }
    cudastream_sp stream_sp;
    cudaStream_t stream;
};

class RandomDMA : public Module
{

public:
    RandomDMA(RandomDMAProps _props);
    virtual ~RandomDMA();
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
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    RandomDMAProps props;
};