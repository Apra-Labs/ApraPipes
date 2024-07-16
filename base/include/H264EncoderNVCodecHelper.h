#pragma once

#include <boost/shared_ptr.hpp>
#include "ImageMetadata.h"
#include "CommonDefs.h"
#include "CudaCommon.h"
#include "H264EncoderNVCodec.h"

/**
 * @brief Helper class for H264 encoding using NVCodec.
 */
class H264EncoderNVCodecHelper
{
public:
    /**
     * @brief Constructor for H264EncoderNVCodecHelper.
     * 
     * @param _bitRateKbps Bit rate in kilobits per second.
     * @param _cuContext CUDA context.
     * @param _gopLength Group of Pictures (GOP) length.
     * @param _frameRate Frame rate.
     * @param _profile Video profile from H264EncoderNVCodecProps::H264CodecProfile enum.
     * @param enableBFrames Enable or disable B-frames.
     */
    H264EncoderNVCodecHelper(uint32_t _bitRateKbps, apracucontext_sp& _cuContext, uint32_t _gopLength, uint32_t _frameRate,H264EncoderNVCodecProps::H264CodecProfile _profile, bool enableBFrames);
    
    /**
     * @brief Constructor for H264EncoderNVCodecHelper with buffer threshold.
     * 
     * @param _bitRateKbps Bit rate in kilobits per second.
     * @param _cuContext CUDA context.
     * @param _gopLength Group of Pictures (GOP) length.
     * @param _frameRate Frame rate.
     * @param _profile Video profile from H264EncoderNVCodecProps::H264CodecProfile enum.
     * @param enableBFrames Enable or disable B-frames.
     * @param _bufferThres Buffer threshold.
     */
    H264EncoderNVCodecHelper(uint32_t _bitRateKbps, apracucontext_sp& _cuContext, uint32_t _gopLength, uint32_t _frameRate,H264EncoderNVCodecProps::H264CodecProfile _profile, bool enableBFrames, uint32_t _bufferThres);
    
    /**
     * @brief Destructor for H264EncoderNVCodecHelper.
     */
    ~H264EncoderNVCodecHelper();

    /**
     * @brief Initialize the encoder.
     * 
     * @param width Frame width.
     * @param height Frame height.
     * @param pitch Frame pitch.
     * @param imageType Type of the image from ImageMetadata::ImageType.
     * @param makeFrame Function to create a frame.
     * @param send Function to send a frame.
     * @return True if initialization was successful, false otherwise.
     */
    bool init(uint32_t width, uint32_t height, uint32_t pitch, ImageMetadata::ImageType imageType, std::function<frame_sp(size_t)> makeFrame, std::function<void(frame_sp& ,frame_sp&)> send);

    /**
     * @brief Process a frame.
     * 
     * @param frame Frame to process.
     * @return True if the frame was processed successfully, false otherwise.
     */
    bool process(frame_sp &frame);

    /**
     * @brief End encoding.
     */
    void endEncode();

    /**
     * @brief Get SPS and PPS data.
     * 
     * @param buffer Pointer to buffer to store SPS and PPS data.
     * @param size Size of the buffer.
     * @param width Width of the frame.
     * @param height Height of the frame.
     * @return True if SPS and PPS data was retrieved successfully, false otherwise.
     */
    bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height);

private:
    /**
     * @brief Internal detail class.
     */
    class Detail;

    /**
     * @brief Shared pointer to the internal detail class.
     */
    boost::shared_ptr<Detail> mDetail;
};