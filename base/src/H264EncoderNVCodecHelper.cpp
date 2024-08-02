#include "H264EncoderNVCodecHelper.h"
#include "ExtFrame.h"
#include "H264EncoderNVCodec.h"
#include "AIPExceptions.h"
#include "Logger.h"
#include "Frame.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/dll.hpp>
#include <thread>
#include <utility>
#include "CudaCommon.h"
#include "nvEncodeAPI.h"

#if defined(_WIN32)
#if defined(_WIN64)
#define NVENC_LIBNAME "nvEncodeAPI64.dll"
#else
#define NVENC_LIBNAME "nvEncodeAPI.dll"
#endif
#else
#define NVENC_LIBNAME "libnvidia-encode.so"
#include <cstring>
static inline bool operator==(const GUID &guid1, const GUID &guid2) {
    return !memcmp(&guid1, &guid2, sizeof(GUID));
}

static inline bool operator!=(const GUID &guid1, const GUID &guid2) {
    return !(guid1 == guid2);
}
#endif

#define DEFAULT_BUFFER_THRESHOLD 30

#define NVENC_API_CALL( nvencAPI )                                                                                 \
    do                                                                                                             \
    {                                                                                                              \
        NVENCSTATUS errorCode = nvencAPI;                                                                          \
        if( errorCode != NV_ENC_SUCCESS)                                                                           \
        {                                                                                                          \
            std::ostringstream errorLog;                                                                           \
            errorLog << #nvencAPI << " returned error " << errorCode;                                              \
			throw AIPException(AIP_FATAL, errorLog.str());														   \
        }                                                                                                          \
    } while (0)																									   			

class NVCodecResources
{
public:
	NVCodecResources(apracucontext_sp& cuContext) : m_cuContext(cuContext),
		m_hEncoder(nullptr),
		m_nFreeOutputBitstreams(0),
		m_nBusyOutputBitstreams(0)
	{
		load2();
	}

	void load()
	{
		m_lib.load(NVENC_LIBNAME,boost::dll::load_mode::search_system_folders);
		if (!m_lib.is_loaded())
		{
			throw AIPException(AIP_FATAL, "NVENC library file is not found. Please ensure NV driver is installed. NV_ENC_ERR_NO_ENCODE_DEVICE");
		}

		{
			boost::function<NVENCSTATUS(uint32_t*)> NvEncodeAPIGetMaxSupportedVersion = m_lib.get<NVENCSTATUS(uint32_t*)>("NvEncodeAPIGetMaxSupportedVersion");
			uint32_t version = 0;
			uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
			NVENC_API_CALL(NvEncodeAPIGetMaxSupportedVersion(&version));
			if (currentVersion > version)
			{
				LOG_ERROR << "NvEncodeAPIGetMaxSupportedVersion " << version << " required " << currentVersion;
				throw AIPException(AIP_FATAL, "Current Driver Version does not support this NvEncodeAPI version, please upgrade driver. NV_ENC_ERR_INVALID_VERSION");
			}
		}
		{
			boost::function<NVENCSTATUS(NV_ENCODE_API_FUNCTION_LIST*)> NvEncodeAPICreateInstance = m_lib.get<NVENCSTATUS(NV_ENCODE_API_FUNCTION_LIST*)>("NvEncodeAPICreateInstance");
			m_nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
			NVENC_API_CALL(NvEncodeAPICreateInstance(&m_nvenc));
			NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
			encodeSessionExParams.device = m_cuContext->getContext();
			encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
			encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
			NVENC_API_CALL(m_nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &m_hEncoder));
		}
	}

	void load2()
	{
		{
			uint32_t version = 0;
			uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
			NVENC_API_CALL(NvEncodeAPIGetMaxSupportedVersion(&version));
			if (currentVersion > version)
			{
				LOG_ERROR << "NvEncodeAPIGetMaxSupportedVersion " << version << " required " << currentVersion;
				throw AIPException(AIP_FATAL, "Current Driver Version does not support this NvEncodeAPI version, please upgrade driver. NV_ENC_ERR_INVALID_VERSION");
			}
			m_nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
			NVENC_API_CALL(NvEncodeAPICreateInstance(&m_nvenc));
			NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
			encodeSessionExParams.device = m_cuContext->getContext();
			encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
			encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
			NVENC_API_CALL(m_nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &m_hEncoder));
		}
	}

	void unload()
	{
		
		 for (auto const &element : registeredResources)
		 {
		 	auto registeredPtr = element.second;
		 	m_nvenc.nvEncUnregisterResource(m_hEncoder, registeredPtr);
		 }
		 registeredResources.clear();

#if defined(_WIN32)
		 while (m_qpCompletionEvent.size())
		 {
		 	auto event = m_qpCompletionEvent.front();
		 	NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
		 	eventParams.completionEvent = event;
		 	m_nvenc.nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
		 	CloseHandle(event);
		 	m_qpCompletionEvent.pop_front();
		 }
#endif
		 while (m_qBitstreamOutputBitstream.size())
		 {
		 	auto buffer = m_qBitstreamOutputBitstream.front();
		 	m_nvenc.nvEncDestroyBitstreamBuffer(m_hEncoder, buffer);
		 	m_qBitstreamOutputBitstream.pop_front();
		 }

		 m_nvenc.nvEncDestroyEncoder(m_hEncoder);

		 m_hEncoder = nullptr;

		 if (m_lib.is_loaded()) m_lib.unload();
	}

	void unlockOutputBitstream(NV_ENC_OUTPUT_PTR outputBitstream)
	{
		 boost::mutex::scoped_lock lock(m_mutex);

		 NVENC_API_CALL(m_nvenc.nvEncUnlockBitstream(m_hEncoder, outputBitstream));

		 m_qBitstreamOutputBitstream.push_back(outputBitstream);

		 m_nFreeOutputBitstreams++;
		 m_not_empty.notify_one();
	}

	~NVCodecResources()
	{
		unload();
	}


public:
	apracucontext_sp m_cuContext;
	boost::dll::shared_library m_lib;
	NV_ENCODE_API_FUNCTION_LIST m_nvenc;
	void *m_hEncoder;

	boost_deque<void *> m_qpCompletionEvent;  // 2threads
	boost_deque<void *> m_qpCompletionEventBusy;  // 2threads
	boost_deque<NV_ENC_OUTPUT_PTR> m_qBitstreamOutputBitstream;  // 3 threads
	boost_deque<NV_ENC_OUTPUT_PTR> m_qBitstreamOutputBitstreamBusy;  // 3 threads
	std::map<void *, NV_ENC_REGISTERED_PTR> registeredResources;

	boost_deque<frame_sp> m_mappedFrames; // 2threads
	boost_deque<NV_ENC_INPUT_PTR> m_mappedResources; // 2threads

	uint32_t m_nFreeOutputBitstreams;
	uint32_t m_nBusyOutputBitstreams;

public:
	boost::condition m_wait_for_output;
	boost::condition m_not_empty;
	boost::mutex m_mutex;

public:	
	boost::object_pool<ExtFrame> frame_opool;
};


class H264EncoderNVCodecHelper::Detail
{
	static GUID asNvidiaGUID(H264EncoderNVCodecProps::H264CodecProfile profileEnum)
	{
		switch (profileEnum)
		{
		case H264EncoderNVCodecProps::BASELINE:
			return NV_ENC_H264_PROFILE_BASELINE_GUID;
		case H264EncoderNVCodecProps::MAIN:
			return NV_ENC_H264_PROFILE_MAIN_GUID;
		case H264EncoderNVCodecProps::HIGH:
			return NV_ENC_H264_PROFILE_HIGH_GUID;
		default:
			throw new AIPException(AIP_NOTEXEPCTED, "Unknown value for H264 Profile!");
		}
	}
public:
	Detail(uint32_t& bitRateKbps, apracucontext_sp& cuContext, uint32_t& gopLength, uint32_t& frameRate,H264EncoderNVCodecProps::H264CodecProfile profile,bool enableBFrames, uint32_t& bufferThres) :
		m_nWidth(0),
		m_nHeight(0),
		m_eBufferFormat(NV_ENC_BUFFER_FORMAT_UNDEFINED),
		m_nEncoderBuffer(0),
		m_nBitRateKbps(bitRateKbps),
		m_nOutSPSPPSPayloadSize(0),
		m_nGopLength(gopLength),
		m_nFrameRate(frameRate),
		m_nProfile(profile),
		m_bEnableBFrames(enableBFrames),
		m_nBufferThres(bufferThres)
	{
		m_nvcodecResources.reset(new NVCodecResources(cuContext));

		m_initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
		m_encodeConfig.version = NV_ENC_CONFIG_VER;
		m_initializeParams.encodeConfig = &m_encodeConfig;
	}

	bool init(uint32_t width, uint32_t height, uint32_t pitch, ImageMetadata::ImageType imageType, std::function<frame_sp(size_t)> _makeFrame, std::function<void(frame_sp&, frame_sp&)> _send)
	{
		m_nWidth = width;
		m_nHeight = height;
		m_nPitch = pitch;

		makeFrame = _makeFrame;
		send = _send;

		switch (imageType)
		{
		case ImageMetadata::YUV420:
			m_eBufferFormat = NV_ENC_BUFFER_FORMAT_IYUV;
			break;
		case ImageMetadata::YUV444:
			m_eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444;
			break;
		case ImageMetadata::BGRA:
			m_eBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
			break;
		case ImageMetadata::RGBA:
			m_eBufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
			break;
		default:
			throw AIPException(AIP_NOTIMPLEMENTED, "Unknown ImageType<" + std::to_string(imageType) + ">");
		}

		// convert imageType to NV_ENC_BUFFER_FORMAT
		createDefaultEncoderParams(&m_initializeParams);
		initializeEncoder();

		// sps pps buffer init
		m_spsppsFrame = makeFrame(NV_MAX_SEQ_HDR_LEN);
		memset(&m_spsppsPayload, 0, sizeof(NV_ENC_SEQUENCE_PARAM_PAYLOAD));
		m_spsppsPayload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
		m_spsppsPayload.inBufferSize = NV_MAX_SEQ_HDR_LEN;
		m_spsppsPayload.spsppsBuffer = m_spsppsFrame->data();
		m_spsppsPayload.outSPSPPSPayloadSize = &m_nOutSPSPPSPayloadSize;
		m_spsppsPayload.spsId;

		m_thread = std::thread(&Detail::processOutput, this);

		return true;
	}

	bool encode(frame_sp &frame)
	{
		auto buffer = frame->data();
		NV_ENC_REGISTERED_PTR registeredPtr = nullptr;
		if (m_nvcodecResources->registeredResources.find(buffer) == m_nvcodecResources->registeredResources.end())
		{
			registeredPtr = RegisterResource(buffer, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, m_nWidth, m_nHeight, m_nPitch, m_eBufferFormat, NV_ENC_INPUT_IMAGE);
			m_nvcodecResources->registeredResources[buffer] = registeredPtr;
		}
		else
		{
			registeredPtr = m_nvcodecResources->registeredResources[buffer];
		}

		NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };

		mapInputResource.registeredResource = registeredPtr;
		NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncMapInputResource(m_nvcodecResources->m_hEncoder, &mapInputResource));

		NV_ENC_OUTPUT_PTR outputBitstream;
		NV_ENC_PIC_PARAMS picParams = {};
		void* event=nullptr;
		{
			boost::mutex::scoped_lock lock(m_nvcodecResources->m_mutex);
			m_nvcodecResources->m_not_empty.wait(lock, boost::bind(&Detail::is_not_empty, this));

			m_nvcodecResources->m_mappedResources.push_back(mapInputResource.mappedResource);
			m_nvcodecResources->m_mappedFrames.push_back(frame);

			outputBitstream = m_nvcodecResources->m_qBitstreamOutputBitstream.front();
			m_nvcodecResources->m_qBitstreamOutputBitstreamBusy.push_back(outputBitstream);
#if defined(_WIN32)
			event = m_nvcodecResources->m_qpCompletionEvent.front();
			m_nvcodecResources->m_qpCompletionEventBusy.push_back(event);
			m_nvcodecResources->m_qpCompletionEvent.pop_front();
#endif
			m_nvcodecResources->m_qBitstreamOutputBitstream.pop_front();
		}

		picParams.version = NV_ENC_PIC_PARAMS_VER;
		picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		picParams.bufferFmt = m_eBufferFormat;
		picParams.inputWidth = m_nWidth;
		picParams.inputHeight = m_nHeight;
		picParams.inputBuffer = mapInputResource.mappedResource;
		picParams.outputBitstream = outputBitstream;
		picParams.completionEvent = event;
		NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncEncodePicture(m_nvcodecResources->m_hEncoder, &picParams));

		{
			boost::mutex::scoped_lock lock(m_nvcodecResources->m_mutex);
			m_nvcodecResources->m_nFreeOutputBitstreams--;
			m_nvcodecResources->m_nBusyOutputBitstreams++;
			m_nvcodecResources->m_wait_for_output.notify_one();
		}

		return true;
	}

	void endEncode()
	{
		return;
#if 0 //AK found dead code
		auto event = m_nvcodecResources->m_qpCompletionEvent.front();
		m_nvcodecResources->m_qpCompletionEventBusy.push_back(event);
		m_nvcodecResources->m_qpCompletionEvent.pop_front();

		NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
		picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
		picParams.completionEvent = event;
		NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncEncodePicture(m_nvcodecResources->m_hEncoder, &picParams));
#endif
	}

	bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height)
	{
		if (!m_spsppsFrame.get())
		{
			return false;
		}

		NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncGetSequenceParams(m_nvcodecResources->m_hEncoder, &m_spsppsPayload));
		buffer = m_spsppsFrame->data();
		size = static_cast<size_t>(m_nOutSPSPPSPayloadSize);

		width = static_cast<int>(m_nWidth);
		height = static_cast<int>(m_nHeight);

		return true;
	}

	~Detail()
	{
		unload();
		m_nvcodecResources.reset();
	}

private:



	void unload()
	{
		{
			boost::mutex::scoped_lock lock(m_nvcodecResources->m_mutex);
			m_bRunning = false;
			m_nvcodecResources->m_wait_for_output.notify_one();
		}

		if(m_thread.joinable())
			m_thread.join();

	}

	void createDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS *pIntializeParams)
	{
		 GUID codecGuid = NV_ENC_CODEC_H264_GUID;

		 memset(pIntializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
		 pIntializeParams->encodeConfig = &m_encodeConfig;
		 memset(&m_encodeConfig, 0, sizeof(NV_ENC_CONFIG));

		 pIntializeParams->encodeConfig->version = NV_ENC_CONFIG_VER;
		 pIntializeParams->version = NV_ENC_INITIALIZE_PARAMS_VER;

		 pIntializeParams->encodeGUID = codecGuid;
		 pIntializeParams->presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
		 pIntializeParams->encodeWidth = m_nWidth;
		 pIntializeParams->encodeHeight = m_nHeight;
		 pIntializeParams->darWidth = m_nWidth;
		 pIntializeParams->darHeight = m_nHeight;
		 pIntializeParams->frameRateNum = m_nFrameRate;
		 pIntializeParams->frameRateDen = 1;
		 pIntializeParams->enablePTD = 1;
		 pIntializeParams->reportSliceOffsets = 0;
		 pIntializeParams->enableSubFrameWrite = 0;
		 pIntializeParams->maxEncodeWidth = m_nWidth;
		 pIntializeParams->maxEncodeHeight = m_nHeight;
		 pIntializeParams->enableMEOnlyMode = false;
		 pIntializeParams->enableOutputInVidmem = false;
#if defined(_WIN32)
		 pIntializeParams->enableEncodeAsync = GetCapabilityValue(codecGuid, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT);
#endif
		 NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER} };
		 m_nvcodecResources->m_nvenc.nvEncGetEncodePresetConfig(m_nvcodecResources->m_hEncoder, codecGuid, pIntializeParams->presetGUID, &presetConfig);
		 memcpy(pIntializeParams->encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
		 pIntializeParams->encodeConfig->frameIntervalP = 1;
		 pIntializeParams->encodeConfig->gopLength = m_nGopLength;// = NVENC_INFINITE_GOPLENGTH;
		 pIntializeParams->encodeConfig->profileGUID = asNvidiaGUID(m_nProfile);
	
	
		 pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
		 m_nencodeParam.capsToQuery = NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE;
		 	if (m_nBitRateKbps)
		 	{
		 		m_encodeConfig.rcParams.averageBitRate = m_nBitRateKbps;
		 	}
		

		
		 m_encodeConfig.rcParams.enableLookahead = m_bEnableBFrames;
		
		 pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

		 if (pIntializeParams->tuningInfo != NV_ENC_TUNING_INFO_LOSSLESS)
		 {
			 pIntializeParams->encodeConfig->rcParams.constQP = { 28, 31, 25 }; //quality params for P, B and I frames
		 }

		 if (pIntializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID)
		 {
		 	if (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
		 	{
		 		pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
		 	}
		 	pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.idrPeriod = pIntializeParams->encodeConfig->gopLength;
		 }
		 else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID)
		 {
		 	pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
		 		(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0;
		 	if (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
		 	{
		 		pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
		 	}
		 	pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = pIntializeParams->encodeConfig->gopLength;
		 }

		 return;
	}

	int GetCapabilityValue(GUID guidCodec, NV_ENC_CAPS capsToQuery)
	{ 
		 NV_ENC_CAPS_PARAM capsParam = { NV_ENC_CAPS_PARAM_VER };
		 capsParam.capsToQuery = capsToQuery;
		 int v=0;
		 NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncGetEncodeCaps(m_nvcodecResources->m_hEncoder, guidCodec, &capsParam, &v));
		 return v;
	}

	void initializeEncoder()
	{
		 NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncInitializeEncoder(m_nvcodecResources->m_hEncoder, &m_initializeParams));

		 m_nEncoderBuffer = m_encodeConfig.frameIntervalP + m_encodeConfig.rcParams.lookaheadDepth + DEFAULT_BUFFER_THRESHOLD;
		 m_nvcodecResources->m_nFreeOutputBitstreams = m_nEncoderBuffer;

		 for (int i = 0; i < m_nEncoderBuffer; i++)
		 {
#if defined(_WIN32)
		 	auto event = CreateEvent(NULL, FALSE, FALSE, NULL);
		 	NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
		 	eventParams.completionEvent = event;
		 	NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncRegisterAsyncEvent(m_nvcodecResources->m_hEncoder, &eventParams));
		 	m_nvcodecResources->m_qpCompletionEvent.push_back(event);
#endif

		 	NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
		 	NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncCreateBitstreamBuffer(m_nvcodecResources->m_hEncoder, &createBitstreamBuffer));
		 	m_nvcodecResources->m_qBitstreamOutputBitstream.push_back(createBitstreamBuffer.bitstreamBuffer);
		 }
	}

	NV_ENC_REGISTERED_PTR RegisterResource(void *pBuffer, NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
		int width, int height, int pitch, NV_ENC_BUFFER_FORMAT bufferFormat, NV_ENC_BUFFER_USAGE bufferUsage)
	{
		 NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
		 registerResource.resourceType = eResourceType;
		 registerResource.resourceToRegister = pBuffer;
		 registerResource.width = width;
		 registerResource.height = height;
		 registerResource.pitch = pitch;
		 registerResource.bufferFormat = bufferFormat;
		 registerResource.bufferUsage = bufferUsage;
		 NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncRegisterResource(m_nvcodecResources->m_hEncoder, &registerResource));

		 return registerResource.registeredResource;
		return nullptr;
	}

private:


	void processOutput()
	{
		 m_bRunning = true;
		 while (true)
		 {
		 	frame_sp inputFrame, outputFrame;
		 	void * event=nullptr;
		 	NV_ENC_OUTPUT_PTR outputBitstream;
		 	NV_ENC_INPUT_PTR mappedResource;
		 	{
		 		boost::mutex::scoped_lock lock(m_nvcodecResources->m_mutex);
		 		m_nvcodecResources->m_wait_for_output.wait(lock, boost::bind(&Detail::is_output_available, this));
		 		if (!m_bRunning)
		 		{
		 			break;
		 		}

		 		outputBitstream = m_nvcodecResources->m_qBitstreamOutputBitstreamBusy.front();
#if defined(_WIN32)
		 		event = m_nvcodecResources->m_qpCompletionEventBusy.front();
#endif
		 		inputFrame = m_nvcodecResources->m_mappedFrames.front();
		 		mappedResource = m_nvcodecResources->m_mappedResources.front();
		 	}
#if defined(_WIN32)
		 	if (WaitForSingleObject(event, 20000) == WAIT_FAILED)
		 	{
		 		throw AIPException(AIP_FATAL, "Failed to encode frame. WaitForSingleObject. <" + std::to_string(NV_ENC_ERR_GENERIC));
		 	}
#endif
		 	NV_ENC_LOCK_BITSTREAM lockBitstreamData = { NV_ENC_LOCK_BITSTREAM_VER };
		 	lockBitstreamData.outputBitstream = outputBitstream;
		 	lockBitstreamData.doNotWait = false;
		 	NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncLockBitstream(m_nvcodecResources->m_hEncoder, &lockBitstreamData));

		 	auto nvCodecResources = m_nvcodecResources;
		 	outputFrame = frame_sp(m_nvcodecResources->frame_opool.construct(lockBitstreamData.bitstreamBufferPtr, lockBitstreamData.bitstreamSizeInBytes),
		 		[&, outputBitstream, nvCodecResources](ExtFrame* pointer) {
		 		nvCodecResources->frame_opool.free(pointer);
		 		nvCodecResources->unlockOutputBitstream(outputBitstream);
		 	});
		 	outputFrame->pictureType = lockBitstreamData.pictureType;

		 	send(inputFrame, outputFrame);

		 	NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncUnmapInputResource(m_nvcodecResources->m_hEncoder, mappedResource));

		 	{
		 		boost::mutex::scoped_lock lock(m_nvcodecResources->m_mutex);
		 		m_nvcodecResources->m_mappedResources.pop_front();
		 		m_nvcodecResources->m_mappedFrames.pop_front();
#if defined(_WIN32)
		 		m_nvcodecResources->m_qpCompletionEventBusy.pop_front();
		 		m_nvcodecResources->m_qpCompletionEvent.push_back(event);
#endif
		 		m_nvcodecResources->m_qBitstreamOutputBitstreamBusy.pop_front();
		 		m_nvcodecResources->m_nBusyOutputBitstreams--;
		 	}
		 }
	}

	/**
	 * @brief Checks if there are free output bitstreams available and allocates more if needed.
	 * 
	 * This function checks the availability of free output bitstreams. If there are no free output bitstreams and the number
	 * of busy streams is below a predefined threshold, it allocates additional output bitstreams to the encoder buffer.
	 * 
	 * @details 
	 * The function first checks the number of busy output bitstreams. If there are no free output bitstreams and the number
	 * of busy streams is below `m_nBufferThres`, it calculates the buffer length to be allocated and doubles the output buffers
	 * by calling `doubleOutputBuffers`. It then logs the allocation and increments the count of free output bitstreams.
	 * If there are free output bitstreams, it logs a message indicating the current state.
	 * 
	 * @return true if there are free output bitstreams available, false otherwise.
	 * 
	 * @note This function modifies the state of `m_nvcodecResources->m_nFreeOutputBitstreams`.
	 * @warning Ensure thread safety when calling this function in a multi-threaded environment.
	 */

	bool is_not_empty() const
	{
		uint32_t busyStreams = m_nvcodecResources->m_nBusyOutputBitstreams;
		if (!m_nvcodecResources->m_nFreeOutputBitstreams && busyStreams < m_nBufferThres)
		{
			uint32_t bufferLength = min(busyStreams, m_nBufferThres - busyStreams);
			doubleOutputBuffers(bufferLength);
			LOG_INFO << "Allocated <" << bufferLength << "> outputbitstreams to the encoder buffer.";
			m_nvcodecResources->m_nFreeOutputBitstreams += bufferLength;
		}
		else
		{
			LOG_INFO << "waiting for free outputbitstream<> busy streams<" << m_nvcodecResources->m_nBusyOutputBitstreams << ">";
		}

		return m_nvcodecResources->m_nFreeOutputBitstreams > 0;
	}

	/**
	 * @brief Checks if there are any busy output bitstreams or if the encoder is not running.
	 * 
	 * This function returns true if there are any busy output bitstreams or if the encoder is not running.
	 * It helps determine if there are any output bitstreams currently being processed or if the encoding process has stopped.
	 * 
	 * @return true if there are busy output bitstreams or the encoder is not running, false otherwise.
	 * 
	 * @note This function does not modify any state.
	 */

	bool is_output_available() const
	{
		return m_nvcodecResources->m_nBusyOutputBitstreams > 0 || !m_bRunning;
	}

	/**
	 * @brief Allocates additional output bitstream buffers.
	 * 
	 * This function allocates a specified number of additional output bitstream buffers and adds them to the encoder buffer.
	 * It uses the NVIDIA Video Codec SDK to create the bitstream buffers and updates the internal queue of output bitstreams.
	 * 
	 * @param bufferLength The number of additional output bitstream buffers to allocate.
	 * 
	 * @note This function is called internally by `is_not_empty` when more buffers are needed.
	 * @warning Ensure that the NVIDIA Video Codec SDK is properly initialized before calling this function.
	 */
	void doubleOutputBuffers(uint32_t bufferLength) const
	{
		for (int i = 0; i < bufferLength; i++)
		{
			NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
			NVENC_API_CALL(m_nvcodecResources->m_nvenc.nvEncCreateBitstreamBuffer(m_nvcodecResources->m_hEncoder, &createBitstreamBuffer));
			m_nvcodecResources->m_qBitstreamOutputBitstream.push_back(createBitstreamBuffer.bitstreamBuffer);
		}
	}

	std::thread m_thread;
	bool m_bRunning;

private:
	uint32_t m_nWidth;
	uint32_t m_nHeight;
	uint32_t m_nPitch;
	NV_ENC_BUFFER_FORMAT m_eBufferFormat;
	uint32_t m_nBitRateKbps;
	uint32_t m_nGopLength ;
	uint32_t m_nFrameRate;
	H264EncoderNVCodecProps::H264CodecProfile m_nProfile;
	bool m_bEnableBFrames;
	uint32_t m_nBufferThres;

	NV_ENC_INITIALIZE_PARAMS m_initializeParams;
	NV_ENC_CONFIG m_encodeConfig;
	int32_t m_nEncoderBuffer;
	NV_ENC_CAPS_PARAM m_nencodeParam;

	std::function<frame_sp(size_t)> makeFrame;
	std::function<void(frame_sp&, frame_sp&)> send;

private:
	frame_sp m_spsppsFrame;
	NV_ENC_SEQUENCE_PARAM_PAYLOAD m_spsppsPayload;
	uint32_t m_nOutSPSPPSPayloadSize;

private:
	boost::shared_ptr<H264EncoderNVCodecProps> mProps;
	boost::shared_ptr<NVCodecResources> m_nvcodecResources;
};

H264EncoderNVCodecHelper::H264EncoderNVCodecHelper(uint32_t _bitRateKbps, apracucontext_sp& _cuContext, uint32_t _gopLength, uint32_t _frameRate, H264EncoderNVCodecProps::H264CodecProfile _profile, bool enableBFrames)
{
	uint32_t _bufferThres = 30;
	mDetail.reset(new Detail(_bitRateKbps, _cuContext,_gopLength,_frameRate,_profile,enableBFrames,_bufferThres));
}

H264EncoderNVCodecHelper::H264EncoderNVCodecHelper(uint32_t _bitRateKbps, apracucontext_sp& _cuContext, uint32_t _gopLength, uint32_t _frameRate, H264EncoderNVCodecProps::H264CodecProfile _profile, bool enableBFrames, uint32_t _bufferThres)
{
	mDetail.reset(new Detail(_bitRateKbps, _cuContext,_gopLength,_frameRate,_profile,enableBFrames,_bufferThres));
}

H264EncoderNVCodecHelper::~H264EncoderNVCodecHelper()
{
	mDetail.reset();
}

bool H264EncoderNVCodecHelper::init(uint32_t width, uint32_t height, uint32_t pitch, ImageMetadata::ImageType imageType, std::function<frame_sp(size_t)> makeFrame, std::function<void(frame_sp&, frame_sp&)> send)
{
	return mDetail->init(width, height, pitch, imageType, makeFrame, send);
}

bool H264EncoderNVCodecHelper::process(frame_sp &frame)
{
	return mDetail->encode(frame);
}

void H264EncoderNVCodecHelper::endEncode()
{
	return mDetail->endEncode();
}

bool H264EncoderNVCodecHelper::getSPSPPS(void*& buffer, size_t& size, int& width, int& height)
{
	return mDetail->getSPSPPS(buffer, size, width, height);
}