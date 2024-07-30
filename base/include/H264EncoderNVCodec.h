#pragma once

#include "Module.h"
#include "CudaCommon.h"

/**
 * @brief Properties for the H264 encoder using NVCodec.
 */
class H264EncoderNVCodecProps : public ModuleProps
{
public:
	/**
	 * @enum H264CodecProfile
	 * @brief Enum representing different H.264 codec profiles.
	 */
	enum H264CodecProfile
	{
		BASELINE, /**< Baseline profile */
		MAIN, /**< Main profile */
		HIGH,  /**< High profile */
	};

	/**
	 * @brief Constructor for H264EncoderNVCodecProps with all parameters.
	 * 
	 * @param _bitRateKbps Bit rate in kilobits per second.
	 * @param _cuContext CUDA context.
	 * @param _gopLength Group of Pictures (GOP) length.
	 * @param _frameRate Frame rate.
	 * @param _vProfile Video profile from H264CodecProfile enum.
	 * @param _enableBFrames Enable or disable B-frames.
	 */
	H264EncoderNVCodecProps(const uint32_t &_bitRateKbps, const apracucontext_sp& _cuContext, const uint32_t &_gopLength,const uint32_t &_frameRate,H264CodecProfile _vProfile,bool _enableBFrames) 
		: cuContext(_cuContext), gopLength(_gopLength), frameRate(_frameRate), bitRateKbps(_bitRateKbps), vProfile(_vProfile), enableBFrames(_enableBFrames)
	{
	}

	/**
	 * @brief Constructor for H264EncoderNVCodecProps with default bit rate.
	 * 
	 * @param _cuContext CUDA context.
	 */
	H264EncoderNVCodecProps(apracucontext_sp& _cuContext) : bitRateKbps(0), cuContext(_cuContext)
	{
	}

	/**
	 * @brief Constructor for H264EncoderNVCodecProps with buffer threshold.
	 * 
	 * @param _bitRateKbps Bit rate in kilobits per second.
	 * @param _cuContext CUDA context.
	 * @param _gopLength Group of Pictures (GOP) length.
	 * @param _frameRate Frame rate.
	 * @param _vProfile Video profile from H264CodecProfile enum.
	 * @param _enableBFrames Enable or disable B-frames.
	 * @param _bufferThres Buffer threshold.
	 */
	H264EncoderNVCodecProps(const uint32_t &_bitRateKbps, const apracucontext_sp& _cuContext, const uint32_t &_gopLength,const uint32_t &_frameRate,H264CodecProfile _vProfile,bool _enableBFrames, uint32_t &_bufferThres) 
		: cuContext(_cuContext), gopLength(_gopLength), frameRate(_frameRate), bitRateKbps(_bitRateKbps), vProfile(_vProfile), enableBFrames(_enableBFrames), bufferThres(_bufferThres)
	{
	}

	H264CodecProfile vProfile = H264EncoderNVCodecProps::BASELINE; /**< Video profile. */
	bool enableBFrames = false; /**< Enable or disable B-frames. */
	uint32_t gopLength = 30; /**< Group of Pictures (GOP) length. */
	uint32_t bitRateKbps = 1000; /**< Bit rate in kilobits per second. */
	uint32_t frameRate = 30; /**< Frame rate. */
	apracucontext_sp cuContext; /**< CUDA context. */
	uint32_t bufferThres = 30; /**< Buffer threshold. */
};

class H264EncoderNVCodec : public Module
{

public:
	H264EncoderNVCodec(H264EncoderNVCodecProps _props);
	virtual ~H264EncoderNVCodec();
	bool init();
	bool term();
	bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height);

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;

	bool mShouldTriggerSOS;
	framemetadata_sp mOutputMetadata;
	std::string mInputPinId;
	std::string mOutputPinId;

	H264EncoderNVCodecProps props;
};
