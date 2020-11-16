#pragma once

#include "FrameMetadata.h"
#include "ImageMetadata.h"
#include <opencv2/opencv.hpp>

class RawImagePlanarMetadata : public FrameMetadata
{
public:
	RawImagePlanarMetadata(MemType _memType) : FrameMetadata(FrameType::RAW_IMAGE_PLANAR, _memType) {}

	RawImagePlanarMetadata(int _width, int _height, ImageMetadata::ImageType _imageType, size_t alignLength, int _depth, MemType _memType = MemType::HOST) : FrameMetadata(FrameType::RAW_IMAGE_PLANAR, _memType)
	{
		size_t _step[4] = {0, 0, 0, 0};

		switch (_imageType)
		{
		case ImageMetadata::YUV444:
			for (auto i = 0; i < 3; i++)
			{
				_step[i] = _width + FrameMetadata::getPaddingLength(_width, alignLength);
			}

			break;
		case ImageMetadata::YUV420:
			_step[0] = _width + FrameMetadata::getPaddingLength(_width, alignLength);
			_step[1] = _step[0] >> 1;
			_step[2] = _step[1];

			break;
		default:
			auto msg = "Unknown image type<" + std::to_string(imageType) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}

		auto elemSize =  ImageMetadata::getElemSize(_depth);
		for (auto i = 0; i < maxChannels; i++)
		{
			_step[i] = _step[i] * elemSize;
		}

		initData(_width, _height, _imageType, _step, _depth);
	}

	RawImagePlanarMetadata(int _width, int _height, ImageMetadata::ImageType _imageType, size_t _step[4], int _depth, MemType _memType = MemType::HOST) : FrameMetadata(FrameType::RAW_IMAGE_PLANAR, _memType)
	{
		initData(_width, _height, _imageType, _step, _depth);
	}

	void reset()
	{
		FrameMetadata::reset();
		// RAW_IMAGE
		for (auto i = 0; i < maxChannels; i++)
		{
			width[i] = NOT_SET_NUM;
			height[i] = NOT_SET_NUM;
			step[i] = NOT_SET_NUM;
		}
		channels = NOT_SET_NUM;
		depth = NOT_SET_NUM;
	}

	void setData(RawImagePlanarMetadata &metadata)
	{
		FrameMetadata::setData(metadata);

		imageType = metadata.imageType;
		channels = metadata.channels;
		for (auto i = 0; i < channels; i++)
		{
			width[i] = metadata.width[i];
			height[i] = metadata.height[i];
			step[i] = metadata.step[i];
		}
		depth = metadata.depth;		

		setDataSize();
	}

	bool isSet()
	{
		return channels != NOT_SET_NUM;
	}

	int getWidth(int channelId)
	{
		return width[channelId];
	}

	int getHeight(int channelId)
	{
		return height[channelId];
	}

	size_t getRowSize(int channelId)
	{
		auto elemSize = ImageMetadata::getElemSize(depth);
		return static_cast<size_t>(width[channelId] * elemSize);
	}

	size_t getStep(int channelId) { return step[channelId]; }
	size_t getNextPtrOffset(int channelId) { return nextPtrOffset[channelId]; }

	size_t getOffset(int channelId, int offsetX, int offsetY)
	{
		if (imageType == ImageMetadata::YUV420 && channelId != 0)
		{
			offsetX = offsetX >> 1;
			offsetY = offsetY >> 1;
		}

		auto elemSize = ImageMetadata::getElemSize(depth);

		return ( step[channelId]*offsetY + (elemSize * offsetX) );
	}

	int getDepth() { return depth; }

	int getChannels() { return channels; }

	ImageMetadata::ImageType getImageType() { return imageType; }

protected:
	void setDataSize()
	{
		dataSize = 0;
		for (auto i = 0; i < channels; i++)
		{
			nextPtrOffset[i] = dataSize;
			dataSize += step[i] * height[i];
		}
	}

	void initData(int _width, int _height, ImageMetadata::ImageType _imageType, size_t _step[4], int _depth)
	{
		channels = 3;
		depth = _depth;
		imageType = _imageType;

		for (auto i = 0; i < channels; i++)
		{
			step[i] = _step[i];
		}

		switch (_imageType)
		{
		case ImageMetadata::YUV444:
			for (auto i = 0; i < channels; i++)
			{
				width[i] = _width;
				height[i] = _height;
			}

			break;
		case ImageMetadata::YUV420:
			width[0] = _width;
			height[0] = _height;
			width[1] = _width >> 1;
			height[1] = _height >> 1;
			width[2] = width[1];
			height[2] = height[1];

			break;
		default:
			auto msg = "Unknown image type<" + std::to_string(imageType) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}

		setDataSize();
	}

	// https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html
	int width[4] = {NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM};
	int height[4] = {NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM};
	int channels = NOT_SET_NUM;
	size_t step[4] = {NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM}; // https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html#a51615ebf17a64c968df0bf49b4de6a3a
	size_t nextPtrOffset[4] = { NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM, NOT_SET_NUM }; 
	int depth = NOT_SET_NUM;

	ImageMetadata::ImageType imageType;

	int maxChannels = 4;
};