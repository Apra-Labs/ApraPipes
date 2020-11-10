#include "JPEGDecoderNVJPEG.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "nvjpeg.h"

class JPEGDecoderNVJPEG::Detail
{
public:
	Detail(JPEGDecoderNVJPEGProps &_props) : props(_props)
	{
		for(auto i = 0; i < 4; i++)
		{
			width[i] = 0;
			height[i] = 0;
		}

		output_format = NVJPEG_OUTPUT_UNCHANGED;
		// currently only MONO and YUV420 supported
		switch(props.imageType)
		{
		case ImageMetadata::UNSET:			
			break;
		case ImageMetadata::MONO:
			output_format = NVJPEG_OUTPUT_Y;
			break;			
		case ImageMetadata::YUV444:
		case ImageMetadata::YUV420:
			output_format = NVJPEG_OUTPUT_YUV;
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::BGRA:
		case ImageMetadata::RGB:
		case ImageMetadata::RGBA:
		default:
			throw AIPException(AIP_FATAL, "Unsupported output imageType<" + std::to_string(props.imageType) + ">");
		}

		auto status = nvjpegCreateSimple(&nv_handle);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegCreateSimple failed.<" + std::to_string(status) + ">");
		}
		status = nvjpegJpegStateCreate(nv_handle, &nv_dec_state);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegJpegStateCreate failed.<" + std::to_string(status) + ">");
		}
	}

	~Detail()
	{		
		auto status = nvjpegJpegStateDestroy(nv_dec_state);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegJpegStateDestroy failed.<" + std::to_string(status) + ">";
		}
		status = nvjpegDestroy(nv_handle);
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegDestroy failed.<" + std::to_string(status) + ">";
		}
	}

	framemetadata_sp setMetadata(void* data, size_t size)
	{			
		nvjpegChromaSubsampling_t  subsampling;

		auto status = nvjpegGetImageInfo(
			nv_handle, const_cast<const unsigned char *>(static_cast<unsigned char *>(data)), size,
			&channels, &subsampling, width, height);
		
		if(status != NVJPEG_STATUS_SUCCESS)
		{
			throw AIPException(AIP_FATAL, "nvjpegGetImageInfo failed. <" + std::to_string(status) + ">");
		}

		nv_image = { { nullptr, nullptr, nullptr, nullptr }, { 0, 0, 0, 0 } };

		if (props.imageType == ImageMetadata::UNSET)
		{
			switch (subsampling)
			{
			case NVJPEG_CSS_420:
				output_format = NVJPEG_OUTPUT_YUV;
				props.imageType = ImageMetadata::YUV420;
				break;
			case NVJPEG_CSS_GRAY:
				output_format = NVJPEG_OUTPUT_Y;
				props.imageType = ImageMetadata::MONO;
				break;
			default:
				throw AIPException(AIP_FATAL, "Unsupported subsampling<" + std::to_string(subsampling) + ">");
			}
		}

		framemetadata_sp metadata;
		switch (props.imageType)
		{
		case ImageMetadata::MONO:
		{
			metadata = framemetadata_sp(new RawImageMetadata(width[0], height[0], props.imageType, CV_8UC1, 512, CV_8U, FrameMetadata::CUDA_DEVICE, true));
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			nv_image.pitch[0] = static_cast<uint32_t>(rawImageMetadata->getStep());
			nextPtrOffset[0] = 0;
			break;
		}
		case ImageMetadata::YUV444:
		case ImageMetadata::YUV420:
		{
			metadata = framemetadata_sp(new RawImagePlanarMetadata(width[0], height[0], props.imageType, 512, CV_8U, FrameMetadata::CUDA_DEVICE));
			auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			for (auto i = 0; i < channels; i++)
			{
				nv_image.pitch[i] = static_cast<uint32_t>(rawImagePlanarMetadata->getStep(i));
				nextPtrOffset[i] = rawImagePlanarMetadata->getNextPtrOffset(i);
			}
			break;
		}
		case ImageMetadata::BGR:
		case ImageMetadata::BGRA:
		case ImageMetadata::RGB:
		case ImageMetadata::RGBA:
		default:
			throw AIPException(AIP_FATAL, "Unsupported output imageType<" + std::to_string(props.imageType) + ">");
		}

		return metadata;
	}


	void fillImage(void* buffer)
	{
		auto pBuffer = static_cast<unsigned char *>(buffer);		

		switch (output_format)
		{
		case NVJPEG_OUTPUT_YUV:
		{
			nv_image.channel[0] = pBuffer;
			nv_image.channel[1] = pBuffer + nextPtrOffset[1];
			nv_image.channel[2] = pBuffer + nextPtrOffset[2];
		}
		break;
		case NVJPEG_OUTPUT_Y:
		{
			nv_image.channel[0] = pBuffer;
		}
		break;
		default:
			throw AIPException(AIP_NOTIMPLEMENTED, "Unsupported subsampling<>" + std::to_string(output_format));
		}
	}

	bool compute(void* buffer, size_t size, void* outBuffer)
	{
		fillImage(outBuffer);
		auto status = nvjpegDecode(nv_handle, nv_dec_state,
								   static_cast<const unsigned char *>(buffer),
								   size, output_format, &nv_image,
								   props.stream);
		
		if (status != NVJPEG_STATUS_SUCCESS)
		{
			LOG_ERROR << "nvjpegDecode failed. <" << status << ">";
			return false;
		}

		return true;
	}

	void getImageSize(int& _width, int& _height)
	{
		_width = width[0];
		_height = height[0];
	}

private:
	JPEGDecoderNVJPEGProps props;

	nvjpegHandle_t nv_handle;
	nvjpegJpegState_t nv_dec_state;
	nvjpegOutputFormat_t output_format;
	nvjpegImage_t nv_image;

	int channels;
	int width[4];
	int height[4];
	size_t nextPtrOffset[4];
};

JPEGDecoderNVJPEG::JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps _props) : Module(TRANSFORM, "JPEGDecoderNVJPEG", _props), mOutputSize(0)
{
	mDetail.reset(new Detail(_props));
	mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::CUDA_DEVICE));
	mOutputPinId = addOutputPin(mOutputMetadata);
}

JPEGDecoderNVJPEG::~JPEGDecoderNVJPEG() {}

bool JPEGDecoderNVJPEG::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins output memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool JPEGDecoderNVJPEG::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}	

	return true;
}

bool JPEGDecoderNVJPEG::init()
{
	if (!Module::init())
	{
		return false;
	}

	if (mOutputMetadata->isSet())
	{
		throw AIPException(AIP_FATAL, "Metadata will be automatically set. Please remove it.");
	}

	return true;
}

bool JPEGDecoderNVJPEG::term()
{
	return Module::term();
}

bool JPEGDecoderNVJPEG::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mOutputSize, mOutputMetadata);

	auto res = mDetail->compute(frame->data(), frame->size(), outFrame->data());
	if (!res)
	{
		return true;
	}

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool JPEGDecoderNVJPEG::processSOS(frame_sp &frame)
{	
	auto hint = mOutputMetadata->getHint();
	mOutputMetadata = mDetail->setMetadata(frame->data(), frame->size());
	mOutputMetadata->setHint(hint);
	mOutputSize = mOutputMetadata->getDataSize();

	return true;
}

bool JPEGDecoderNVJPEG::shouldTriggerSOS()
{
	return mOutputSize == 0;
}

bool JPEGDecoderNVJPEG::processEOS(string& pinId)
{
	mOutputSize = 0;
	return true;
}

void JPEGDecoderNVJPEG::getImageSize(int& width, int& height)
{
	mDetail->getImageSize(width, height);
}