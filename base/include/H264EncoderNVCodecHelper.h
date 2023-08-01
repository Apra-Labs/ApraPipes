#pragma once

#include <boost/shared_ptr.hpp>
#include "ImageMetadata.h"
#include "CommonDefs.h"
#include "CudaCommon.h"
#include "H264EncoderNVCodec.h"
class H264EncoderNVCodecHelper
{
public:
    H264EncoderNVCodecHelper(uint32_t _bitRateKbps, apracucontext_sp& _cuContext, uint32_t _gopLength, uint32_t _frameRate,H264EncoderNVCodecProps::H264CodecProfile _profile, bool enableBFrames);
    ~H264EncoderNVCodecHelper();

    bool init(uint32_t width, uint32_t height, uint32_t pitch, ImageMetadata::ImageType imageType, void* inputFrameBuffer, std::function<frame_sp(size_t)> makeFrame, std::function<void(frame_sp& ,frame_sp&)> send);

    bool process(frame_sp &frame);
    void endEncode();
    bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};