#pragma once

#include "FrameMetadata.h"
#include "ImageMetadata.h"
#include <opencv2/opencv.hpp>

// Interleaved

class RawImageMetadata : public FrameMetadata
{
public:
	RawImageMetadata() : FrameMetadata(FrameType::RAW_IMAGE) {}
	RawImageMetadata(std::string _hint) : FrameMetadata(FrameType::RAW_IMAGE, _hint) {}
	RawImageMetadata(MemType _memType) : FrameMetadata(FrameType::RAW_IMAGE, _memType) {}

	RawImageMetadata(int _width, int _height, int _channels, int _type, size_t _step, int _depth) : FrameMetadata(FrameType::RAW_IMAGE, FrameMetadata::HOST)
	{
		if (_channels == 1)
		{
			imageType = ImageMetadata::MONO;
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Use different constructor. This constructor will be deprecated in the future.");
		}

		width = _width;
		height = _height;
		channels = _channels;
		type = _type;
		step = _step;
		depth = _depth;
		setDataSize();
	}

	RawImageMetadata(int _width, int _height, ImageMetadata::ImageType _imageType, int _type, size_t alignLength, int _depth, MemType _memType, bool computeStep=false) : FrameMetadata(FrameType::RAW_IMAGE, _memType)
	{
		if (!computeStep)
		{
			auto _step = alignLength;
			initData(_width, _height, _imageType, _type, _step, _depth);
			return;
		}

		size_t _step = 0;
		switch (_imageType)
		{
		case ImageMetadata::MONO:
			_step = _width;
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
			_step = _width * 3;
			break;
		case ImageMetadata::YUV411_I:
			_step = ((_width * 3) >> 1);
			break;
		case ImageMetadata::RGBA:
		case ImageMetadata::BGRA:
			_step = _width * 4;
			break;
		default:
			auto msg = "Unknown image type<" + std::to_string(imageType) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}

		_step = _step + FrameMetadata::getPaddingLength(_step, alignLength);
		_step = _step * ImageMetadata::getElemSize(_depth);

		initData(_width, _height, _imageType, _type, _step, _depth);
	}

	void reset()
	{
		FrameMetadata::reset();
		// RAW_IMAGE
		width = NOT_SET_NUM;
		height = NOT_SET_NUM;
		channels = NOT_SET_NUM;
		type = NOT_SET_NUM;
		step = NOT_SET_NUM;
		depth = NOT_SET_NUM;
	}

	bool isSet()
	{
		return width != NOT_SET_NUM;
	}

	void setData(cv::Mat &img)
	{
		// applicable only for rgba, mono
		width = img.cols;
		height = img.rows;
		channels = img.channels();
		type = img.type();
		step = img.step;
		depth = img.depth();
		setDataSize();
	}

	void setData(RawImageMetadata &metadata)
	{
		FrameMetadata::setData(metadata);

		imageType = metadata.imageType;
		width = metadata.width;
		height = metadata.height;
		channels = metadata.channels;
		type = metadata.type;
		step = metadata.step;
		depth = metadata.depth;

		setDataSize();
	}

	int getWidth()
	{
		return width;
	}

	size_t getRowSize()
	{
		float multiple = 1;
		switch (imageType)
		{
		case ImageMetadata::MONO:
			multiple = 1;
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
			multiple = 3;
			break;
		case ImageMetadata::YUV411_I:
			multiple = 1.5;
			break;
		case ImageMetadata::RGBA:
		case ImageMetadata::BGRA:
			multiple = 4;
			break;
		default:
			auto msg = "Unknown image type<" + std::to_string(imageType) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}

		auto elemSize = ImageMetadata::getElemSize(depth);

		return static_cast<size_t>(width*multiple*elemSize);
	}

	int getHeight()
	{
		return height;
	}

	int getType() { return type; }

	size_t getStep() { return step; }

	size_t getOffset(int offsetX, int offsetY) 
	{
		auto elemSize = ImageMetadata::getElemSize(depth);

		return (step*offsetY + (elemSize*offsetX*channels) );
	}

	int getChannels() { return channels; }

	int getDepth() { return depth; }

	ImageMetadata::ImageType getImageType() { return imageType; }

protected:
	void setDataSize()
	{		
		dataSize = height * step ;
	}

	void initData(int _width, int _height, ImageMetadata::ImageType _imageType, int _type, size_t _step, int _depth, MemType _memType = MemType::HOST)
	{
		imageType = _imageType;
		width = _width;
		height = _height;
		type = _type;
		step = _step;
		depth = _depth;
		setDataSize();

		switch (imageType)
		{
		case ImageMetadata::MONO:
			channels = 1;
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
		case ImageMetadata::YUV411_I:
			channels = 3;
			break;
		case ImageMetadata::RGBA:
		case ImageMetadata::BGRA:
			channels = 4;
			break;
		default:
			auto msg = "Unknown image type<" + std::to_string(imageType) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}
	}

	// https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html
	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
	int channels = NOT_SET_NUM;
	int type = NOT_SET_NUM;
	size_t step = NOT_SET_NUM; // https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html#a51615ebf17a64c968df0bf49b4de6a3a
	int depth = NOT_SET_NUM;

	ImageMetadata::ImageType imageType = ImageMetadata::ImageType::MONO;
};