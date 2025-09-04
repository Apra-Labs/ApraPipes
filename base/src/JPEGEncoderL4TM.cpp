#include "JPEGEncoderL4TMHelper.h"
#include "JPEGEncoderL4TM.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "RawImagePlanarMetadata.h"
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include "AIPExceptions.h"

#define MSDK_ALIGN16(value)                      (((value + 15) >> 4) << 4) // round up to a multiple of 16
#define MSDK_ALIGN32(X) (((uint32_t)((X)+31)) & (~ (uint32_t)31))
#define MSDK_ALIGN7(X) (((uint32_t)((X)+7)) & (~ (uint32_t)7))

class JPEGEncoderL4TM::Detail
{

public:
	Detail(JPEGEncoderL4TMProps& props): color_space(JCS_YCbCr)
	{
		mProps = JPEGEncoderL4TMProps(props);
	}

	~Detail()
	{
	}

	void setImageMetadata(framemetadata_sp& metadata) 
	{
		mMetadata = metadata;
		init(metadata);
	}

	void setOutputMetadata(framemetadata_sp& metadata) 
	{
		mOutputMetadata = metadata;
	}

	framemetadata_sp getOutputMetadata()
	{
		return mOutputMetadata;
	}

	void copyYUV420PlanarToContiguous(const unsigned char* base, unsigned char* dst, RawImagePlanarMetadata* rim)
	{
		// const int width = rim->getWidth(0);
		// const int height = rim->getHeight(0);
		// const size_t stepY = rim->getStep(0);
		// const size_t stepU = rim->getStep(1);
		// const size_t stepV = rim->getStep(2);
		// const size_t offY = rim->getNextPtrOffset(0);
		// const size_t offU = rim->getNextPtrOffset(1);
		// const size_t offV = rim->getNextPtrOffset(2);
		// const unsigned char* srcY = base + offY;
		// const unsigned char* srcU = base + offU;
		// const unsigned char* srcV = base + offV;
		// unsigned char* dstY = dst;
		// unsigned char* dstU = dst + width * height;
		// unsigned char* dstV = dstU + (width * height) / 4;
		// // Y
		// for (int r = 0; r < height; r++)
		// {
		// 	memcpy(dstY + r * width, srcY + r * stepY, width);
		// }
		// // U and V are half resolution
		// const int uvWidth = width / 2;
		// const int uvHeight = height / 2;
		// for (int r = 0; r < uvHeight; r++)
		// {
		// 	memcpy(dstU + r * uvWidth, srcU + r * stepU, uvWidth);
		// 	memcpy(dstV + r * uvWidth, srcV + r * stepV, uvWidth);
		// }
	}

	size_t compute(frame_sp& inFrame, frame_sp& frame)
	{
		auto in_buf = static_cast<const unsigned char*>(inFrame->data());

		if(color_space == JCS_YCbCr)
		{
			if (mMetadata)
			{
				switch (mMetadata->getFrameType())
				{
				case FrameMetadata::RAW_IMAGE:
				{
					auto rim = FrameMetadataFactory::downcast<RawImageMetadata>(mMetadata);
					if (rim->getImageType() == ImageMetadata::NV12)
					{
						convertNV12toYUV420((unsigned char*)inFrame->data(), (unsigned char*)dummyBuffer.get(), rim->getWidth(), rim->getHeight(), rim->getStep());
						in_buf = dummyBuffer.get();
					}
					else
					{
						memcpy(dummyBuffer.get(), inFrame->data(), inFrame->size());
						in_buf = dummyBuffer.get();
					}
					break;
				}
				case FrameMetadata::RAW_IMAGE_PLANAR:
				{
					auto rim = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mMetadata);
					if (rim->getImageType() == ImageMetadata::NV12)
					{
						convertNV12PlanarToYUV420((unsigned char*)inFrame->data(), (unsigned char*)dummyBuffer.get(), rim);
						in_buf = dummyBuffer.get();
					}
					else if (rim->getImageType() == ImageMetadata::YUV420)
					{
						// copyYUV420PlanarToContiguous((unsigned char*)inFrame->data(), (unsigned char*)dummyBuffer.get(), rim);
						in_buf = dummyBuffer.get();
					}
					else
					{
						// Unexpected planar format; fallback copy
						memcpy(dummyBuffer.get(), inFrame->data(), std::min((size_t)rim->getStep(0) * rim->getHeight(0), mDataSize));
						in_buf = dummyBuffer.get();
					}
					break;
				}
				default:
					memcpy(dummyBuffer.get(), inFrame->data(), inFrame->size());
					in_buf = dummyBuffer.get();
					break;
				}
			}
		}		

		size_t outLength = mDataSize;
		auto out_buf = (unsigned char*)frame->data();	
		encHelper->encode(in_buf, &out_buf, outLength);
		
		return outLength;
	}

	size_t getDataSize()
	{
		return mDataSize;
	}

	bool shouldTriggerSOS()
	{
		return !mMetadata.get();
	}

	void resetMetadata()
	{
		mMetadata.reset();
	}

	bool validateMetadata(framemetadata_sp &metadata, std::string id)
	{
		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		if (rawImageMetadata->getChannels() != 1 && rawImageMetadata->getChannels() != 3)
		{
			LOG_ERROR << "<" << id << ">:: RAW_IMAGE CHANNELS IS EXPECTED TO BE 1 or 3";
			return false;
		}

		if(rawImageMetadata->getImageType() != ImageMetadata::MONO && rawImageMetadata->getImageType() != ImageMetadata::RGB && rawImageMetadata->getImageType() != ImageMetadata::NV12)
		{
			LOG_ERROR << "<" << id << ">:: ImageType is expected to be MONO, RGB, or NV12";
			return false;
		}

		if (rawImageMetadata->getStep()*mProps.scale != MSDK_ALIGN32(rawImageMetadata->getWidth()*rawImageMetadata->getChannels()*mProps.scale))
		{
			LOG_ERROR << "<" << id << ">:: RAW_IMAGE STEP IS EXPECTED TO BE 32 BIT ALIGNED<>" << rawImageMetadata->getWidth() << "<>" << mProps.scale;
			return false;
		}

		return true;
	}
	std::string outputPinId;

private:

	void convertNV12toYUV420(const unsigned char* nv12Data, unsigned char* yuv420Data, uint32_t width, uint32_t height, uint32_t stride)
	{
		// NV12 format: Y plane followed by interleaved UV plane
		// YUV420 format: Y plane, U plane, V plane
		
		uint32_t ySize = width * height;
		uint32_t uvSize = ySize / 4;
		
		// Copy Y plane (same in both formats)
		// Handle stride by copying row by row
		const unsigned char* srcY = nv12Data;
		unsigned char* dstY = yuv420Data;
		for (uint32_t row = 0; row < height; row++)
		{
			memcpy(dstY, srcY, width);
			srcY += stride;
			dstY += width;
		}
		
		// Convert interleaved UV to separate U and V planes
		const unsigned char* uvPlane = nv12Data + stride * height;
		unsigned char* uPlane = yuv420Data + ySize;
		unsigned char* vPlane = uPlane + uvSize;
		
		// Handle UV plane with stride
		uint32_t uvStride = stride;
		uint32_t uvWidth = width / 2;
		uint32_t uvHeight = height / 2;
		
		for (uint32_t row = 0; row < uvHeight; row++)
		{
			for (uint32_t col = 0; col < uvWidth; col++)
			{
				uint32_t srcIdx = row * uvStride + col * 2;
				uint32_t dstIdx = row * uvWidth + col;
				uPlane[dstIdx] = uvPlane[srcIdx];     // U component
				vPlane[dstIdx] = uvPlane[srcIdx + 1]; // V component
			}
		}
	}

	void convertNV12PlanarToYUV420(const unsigned char* base, unsigned char* yuv420Data, RawImagePlanarMetadata* rim)
	{
		// From planar NV12 (channel 0: Y, channel 1: interleaved UV) to contiguous YUV420 planar
		const int width = rim->getWidth(0);
		const int height = rim->getHeight(0);
		const size_t stepY = rim->getStep(0);
		const size_t stepUV = rim->getStep(1);
		const size_t offY = rim->getNextPtrOffset(0);
		const size_t offUV = rim->getNextPtrOffset(1);
		const unsigned char* srcY = base + offY;
		const unsigned char* srcUV = base + offUV;
		unsigned char* dstY = yuv420Data;
		const int ySize = width * height;
		unsigned char* uPlane = yuv420Data + ySize;
		unsigned char* vPlane = uPlane + (ySize / 4);
		// Copy Y plane row by row
		for (int r = 0; r < height; r++)
		{
			memcpy(dstY + r * width, srcY + r * stepY, width);
		}
		// Deinterleave UV
		const int uvWidth = width / 2;
		const int uvHeight = height / 2;
		for (int r = 0; r < uvHeight; r++)
		{
			const unsigned char* srcRow = srcUV + r * stepUV;
			unsigned char* dstURow = uPlane + r * uvWidth;
			unsigned char* dstVRow = vPlane + r * uvWidth;
			for (int c = 0; c < uvWidth; c++)
			{
				const int s = c * 2;
				dstURow[c] = srcRow[s];
				dstVRow[c] = srcRow[s + 1];
			}
		}
	}

	void init(framemetadata_sp& metadata)
	{		
		if(!metadata)
		{
			LOG_INFO << "Empty Metadata";
			return;
		}
		auto frameType = metadata->getFrameType();
		if(frameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			
			if(rawImageMetadata->getImageType() == ImageMetadata::RGB)
			{
				color_space = JCS_RGB;
			}
			else if(rawImageMetadata->getImageType() == ImageMetadata::NV12 || rawImageMetadata->getImageType() == ImageMetadata::YUV420)
			{
				color_space = JCS_YCbCr;
			}
		
			encHelper.reset(new JPEGEncoderL4TMHelper(mProps.quality));
			uint32_t width = rawImageMetadata->getWidth();
			uint32_t height = rawImageMetadata->getHeight();
			uint32_t strideParam = (color_space == JCS_YCbCr) ? width : rawImageMetadata->getStep();
			encHelper->init(width, height, strideParam, color_space, mProps.scale);
			size_t dummyBufferLength = (color_space == JCS_YCbCr) ? static_cast<size_t>(width) * height * 3 / 2 : rawImageMetadata->getDataSize();
			if (color_space == JCS_YCbCr)
			{
				dummyBuffer.reset(new unsigned char[dummyBufferLength]);
				memset(dummyBuffer.get(), 128, dummyBufferLength);
			}
			mDataSize = dummyBufferLength;
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rim = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			if(rim->getImageType() == ImageMetadata::RGB)
			{
				color_space = JCS_RGB;
			}
			else if(rim->getImageType() == ImageMetadata::NV12 || rim->getImageType() == ImageMetadata::YUV420)
			{
				color_space = JCS_YCbCr;
			}
			encHelper.reset(new JPEGEncoderL4TMHelper(mProps.quality));
			uint32_t width = rim->getWidth(0);
			uint32_t height = rim->getHeight(0);
			uint32_t strideParam = (color_space == JCS_YCbCr) ? width : static_cast<uint32_t>(rim->getStep(0));
			encHelper->init(width, height, strideParam, color_space, mProps.scale);
			// tight YUV420 buffer (no padding)
			size_t dummyBufferLength = (color_space == JCS_YCbCr) ? static_cast<size_t>(width) * height * 3 / 2 : rim->getDataSizeByChannel(0);
			if (color_space == JCS_YCbCr)
			{
				dummyBuffer.reset(new unsigned char[dummyBufferLength]);
				memset(dummyBuffer.get(), 128, dummyBufferLength);
			}
			mDataSize =  dummyBufferLength;
		}
	}	

	boost::shared_ptr<unsigned char[]> dummyBuffer;
	boost::shared_ptr<JPEGEncoderL4TMHelper> encHelper;

	framemetadata_sp mMetadata;		
	framemetadata_sp mOutputMetadata;	
	size_t mDataSize;	

	J_COLOR_SPACE color_space;

	JPEGEncoderL4TMProps mProps;
};

JPEGEncoderL4TM::JPEGEncoderL4TM(JPEGEncoderL4TMProps props) : Module(TRANSFORM, "JPEGEncoderL4TM", props)
{
	mDetail.reset(new Detail(props));
}

JPEGEncoderL4TM::~JPEGEncoderL4TM() {}

bool JPEGEncoderL4TM::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();

	if (metadata->isSet())
	{
		// return mDetail->validateMetadata(metadata, getId());
	}

	return true;
}

bool JPEGEncoderL4TM::validateOutputPins()
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

bool JPEGEncoderL4TM::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getOutputMetadataByType(FrameMetadata::ENCODED_IMAGE);
	mDetail->setOutputMetadata(metadata);
	mDetail->outputPinId = getOutputPinIdByType(FrameMetadata::ENCODED_IMAGE);

	return true;
}

bool JPEGEncoderL4TM::term()
{
	return Module::term();
}

bool JPEGEncoderL4TM::process(frame_container& frames)
{
	// auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	// if (isFrameEmpty(frame))
	// {
	// 	return true;
	// }

	auto frame = frames.cbegin()->second;

	auto metadata = mDetail->getOutputMetadata();
	auto bufferFrame = makeFrame(mDetail->getDataSize());		

	size_t frameLength = mDetail->compute(frame, bufferFrame);
		
	auto outFrame = makeFrame(bufferFrame, frameLength, mDetail->outputPinId);

	frames.insert(make_pair(mDetail->outputPinId, outFrame));
	send(frames);
	return true;
}

bool JPEGEncoderL4TM::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setImageMetadata(metadata);
	return true;
}

bool JPEGEncoderL4TM::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool JPEGEncoderL4TM::processEOS(string& pinId)
{
	mDetail->resetMetadata();
	return true;
}