#include "JPEGEncoderNVJPEG.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "nvjpeg.h"

class JPEGEncoderNVJPEG::Detail
{
public:
	Detail(JPEGEncoderNVJPEGProps &_props) : props(_props), width(0), height(0)
	{
		auto status = nvjpegCreateSimple(&nv_handle);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegCreateSimple failed.<" + std::to_string(status) + ">");
		}
		status = nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncoderStateCreate failed.<" + std::to_string(status) + ">");
		}
		status = nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncoderParamsCreate failed.<" + std::to_string(status) + ">");
		}
	}

	~Detail()
	{
		auto status = nvjpegEncoderParamsDestroy(nv_enc_params);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegEncoderParamsDestroy failed.<" + std::to_string(status) + ">";
		}
		status = nvjpegEncoderStateDestroy(nv_enc_state);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegEncoderStateDestroy failed.<" + std::to_string(status) + ">";
		}
		status = nvjpegDestroy(nv_handle);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegDestroy failed.<" + std::to_string(status) + ">";
		}
	}

	void getImageSize(int &_width, int &_height)
	{
		_width = width;
		_height = height;
	}

	size_t setMetadata(framemetadata_sp &metadata)
	{
		nv_image = { {nullptr, nullptr, nullptr, nullptr}, {0, 0, 0, 0} };

		for (auto i = 0; i < 4; i++)
		{
			nextPtrOffset[i] = 0;
		}

		input_format = NVJPEG_INPUT_BGRI;

		if (metadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE)
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			width = rawImageMetadata->getWidth();
			height = rawImageMetadata->getHeight();
			nv_image.pitch[0] = static_cast<uint32_t>(rawImageMetadata->getStep());
			switch (rawImageMetadata->getImageType())
			{
			case ImageMetadata::ImageType::MONO:
				subsampling = NVJPEG_CSS_GRAY;
				isYUV = true;
				break;
			case ImageMetadata::ImageType::RGB:
				isYUV = false;
				subsampling = NVJPEG_CSS_420;
				input_format = NVJPEG_INPUT_RGBI;
				break;
			case ImageMetadata::ImageType::BGR:
			case ImageMetadata::ImageType::RGBA:
			case ImageMetadata::ImageType::BGRA:
			default:
				throw AIPException(AIP_NOTIMPLEMENTED, "Unknown imageType<" + std::to_string(rawImageMetadata->getImageType()) + ">");
			}
		}
		else if (metadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE_PLANAR)
		{
			auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			width = rawImagePlanarMetadata->getWidth(0);
			height = rawImagePlanarMetadata->getHeight(0);
			auto channels = rawImagePlanarMetadata->getChannels();
			for (auto i = 0; i < channels; i++)
			{
				nv_image.pitch[i] = static_cast<uint32_t>(rawImagePlanarMetadata->getStep(i));
				nextPtrOffset[i] = rawImagePlanarMetadata->getNextPtrOffset(i);
			}
			switch (rawImagePlanarMetadata->getImageType())
			{
			case ImageMetadata::ImageType::YUV420:
				subsampling = NVJPEG_CSS_420;
				isYUV = true;
				break;
			case ImageMetadata::ImageType::YUV444:
				subsampling = NVJPEG_CSS_444;
				isYUV = true;
				break;
			default:
				throw AIPException(AIP_NOTIMPLEMENTED, "Unknown imageType<" + std::to_string(rawImagePlanarMetadata->getImageType()) + ">");
			}
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
		}

		// sample input parameters
		auto status = nvjpegEncoderParamsSetQuality(nv_enc_params, props.quality, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncoderParamsSetQuality failed.<" + std::to_string(status) + ">");
		}
		status = nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncoderParamsSetOptimizedHuffman failed.<" + std::to_string(status) + ">");
		}

		status = nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, subsampling, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncoderParamsSetSamplingFactors failed.<" + std::to_string(status) + ">");
		}

		size_t max_stream_length = 0;
		status = nvjpegEncodeGetBufferSize(
			nv_handle,
			nv_enc_params,
			width,
			height,
			&max_stream_length);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegEncodeGetBufferSize failed.<" + std::to_string(status) + ">");
		}

		return max_stream_length;
	}

	void fillImage(void *buffer)
	{
		auto pBuffer = static_cast<unsigned char *>(buffer);

		if (!isYUV)
		{
			// interleaved
			nv_image.channel[0] = pBuffer;
			return;
		}

		switch (subsampling)
		{
		case NVJPEG_CSS_444:
		case NVJPEG_CSS_420:
		{
			nv_image.channel[0] = pBuffer;
			nv_image.channel[1] = pBuffer + nextPtrOffset[1];
			nv_image.channel[2] = pBuffer + nextPtrOffset[2];
		}
		break;
		case NVJPEG_CSS_GRAY:
		{
			nv_image.channel[0] = pBuffer;
		}
		break;
		case NVJPEG_CSS_440:
		case NVJPEG_CSS_422:
		case NVJPEG_CSS_411:
		case NVJPEG_CSS_410:
		default:
			throw AIPException(AIP_NOTIMPLEMENTED, "Unsupported subsampling<>" + std::to_string(subsampling));
		}
	}

	bool compute(void *buffer, void *outBuffer, size_t &length)
	{
		fillImage(buffer);

		auto status = NVJPEG_STATUS_SUCCESS;
		if (isYUV)
		{
			status = nvjpegEncodeYUV(nv_handle,
				nv_enc_state,
				nv_enc_params,
				&nv_image,
				subsampling,
				width,
				height,
				props.stream);
		}
		else
		{
			status = nvjpegEncodeImage(nv_handle,
				nv_enc_state,
				nv_enc_params,
				&nv_image,
				input_format,
				width,
				height,
				props.stream);
		}

		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "Encode failed. <" << status << ">";
		}

		status = nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, static_cast<uint8_t *>(outBuffer), &length, props.stream);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegEncodeRetrieveBitstream failed. <" << status << ">";
		}

		//synchronize
		auto syncStatus = cudaStreamSynchronize(props.stream);
		if (syncStatus != cudaSuccess)
		{
			LOG_ERROR << "cudaStreamSynchronize failed.<" << status << ">";
		}

		return true;
	}

private:
	JPEGEncoderNVJPEGProps props;

	nvjpegHandle_t nv_handle;
	nvjpegEncoderState_t nv_enc_state;
	nvjpegEncoderParams_t nv_enc_params;
	nvjpegChromaSubsampling_t subsampling;
	nvjpegInputFormat_t input_format;
	nvjpegImage_t nv_image;

	int width;
	int height;
	size_t nextPtrOffset[4];
	bool isYUV;
};

JPEGEncoderNVJPEG::JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps _props) : Module(TRANSFORM, "JPEGEncoderNVJPEG", _props), mMaxStreamLength(0)
{
	mDetail.reset(new Detail(_props));
	mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	mOutputPinId = addOutputPin(mOutputMetadata);
}

JPEGEncoderNVJPEG::~JPEGEncoderNVJPEG() {}

bool JPEGEncoderNVJPEG::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool JPEGEncoderNVJPEG::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool JPEGEncoderNVJPEG::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	if (metadata->isSet())
	{
		mMaxStreamLength = mDetail->setMetadata(metadata);
	}

	return true;
}

bool JPEGEncoderNVJPEG::term()
{
	return Module::term();
}

bool JPEGEncoderNVJPEG::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;

	auto buffer = makeBuffer(mMaxStreamLength, mOutputMetadata->getMemType());

	size_t frameLength = mMaxStreamLength;
	mDetail->compute(frame->data(), buffer->data(), frameLength);

	auto outFrame = makeFrame(buffer, frameLength, mOutputMetadata);

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool JPEGEncoderNVJPEG::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mMaxStreamLength = mDetail->setMetadata(metadata);

	return true;
}

bool JPEGEncoderNVJPEG::shouldTriggerSOS()
{
	return mMaxStreamLength == 0;
}

bool JPEGEncoderNVJPEG::processEOS(string &pinId)
{
	mMaxStreamLength = 0;
	return true;
}

void JPEGEncoderNVJPEG::getImageSize(int &width, int &height)
{
	mDetail->getImageSize(width, height);
}