#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FrameMetadataFactory.h"

BOOST_AUTO_TEST_SUITE(imagemetadata_tests)

BOOST_AUTO_TEST_CASE(rawimage_mono)
{
	int width = 1920;
	int height = 1080;
	int channels = 1;
	size_t step = 2048;
	int type = CV_8UC1;
	int depth = CV_8U;

	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, channels, type, step, depth));

	auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	BOOST_TEST(ptr->getWidth() == width);
	BOOST_TEST(ptr->getHeight() == height);
	BOOST_TEST(ptr->getChannels() == channels);
	BOOST_TEST(ptr->getStep() == step);
	BOOST_TEST(ptr->getType() == type);
	BOOST_TEST(ptr->getDataSize() == step * height);
}

BOOST_AUTO_TEST_CASE(rawimage_rgb)
{
	int width = 1920;
	int height = 1080;
	int channels = 3;
	size_t step = 6144;
	int type = CV_8UC3;
	int depth = CV_8U;

	{
		auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, type, step, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		BOOST_TEST(ptr->getWidth() == width);
		BOOST_TEST(ptr->getHeight() == height);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep() == step);
		BOOST_TEST(ptr->getType() == type);
		BOOST_TEST(ptr->getDataSize() == step * height);
	}

	{
		auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, type, 512, depth, FrameMetadata::HOST, true));

		auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		BOOST_TEST(ptr->getWidth() == width);
		BOOST_TEST(ptr->getHeight() == height);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep() == step);
		BOOST_TEST(ptr->getType() == type);
		BOOST_TEST(ptr->getDataSize() == step * height);
	}
}

BOOST_AUTO_TEST_CASE(rawimage_yuv411)
{
	int width = 1920;
	int height = 1080;
	int channels = 3;
	size_t step = 3072;
	int type = CV_8UC3;
	int depth = CV_8U; 

	{
		auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::YUV411_I, type, step, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		BOOST_TEST(ptr->getWidth() == width);
		BOOST_TEST(ptr->getHeight() == height);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep() == step);
		BOOST_TEST(ptr->getType() == type);
		BOOST_TEST(ptr->getDataSize() == step * height);
	}

	{
		auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::YUV411_I, type, 512, depth, FrameMetadata::HOST, true));

		auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		BOOST_TEST(ptr->getWidth() == width);
		BOOST_TEST(ptr->getHeight() == height);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep() == step);
		BOOST_TEST(ptr->getType() == type);
		BOOST_TEST(ptr->getDataSize() == step * height);
	}
}

BOOST_AUTO_TEST_CASE(rawimageplanar_yuv420)
{
	int width = 1920;
	int height = 1080;
	size_t step[4] = { 2048, 1024, 1024 };
	size_t offset = static_cast<size_t>(2048 * height);
	size_t dataSize = step[0] * height;
	dataSize += dataSize >> 1;
	size_t nextPtrOffset[4] = { 0, offset, offset + (offset >> 2), 0 };
	int channels = 3;
	int depth = CV_8U;

	{
		auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV420, step, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		BOOST_TEST(ptr->getWidth(0) == width);
		BOOST_TEST(ptr->getHeight(0) == height);
		BOOST_TEST(ptr->getWidth(1) == width / 2);
		BOOST_TEST(ptr->getHeight(1) == height / 2);
		BOOST_TEST(ptr->getWidth(2) == width / 2);
		BOOST_TEST(ptr->getHeight(2) == height / 2);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep(0) == step[0]);
		BOOST_TEST(ptr->getStep(1) == step[1]);
		BOOST_TEST(ptr->getStep(2) == step[2]);		
		BOOST_TEST(ptr->getDataSize() == dataSize);
		for (auto i = 0; i < 3; i++)
		{
			BOOST_TEST(ptr->getNextPtrOffset(i) == nextPtrOffset[i]);
		}
	}

	{
		auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV420, 512, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		BOOST_TEST(ptr->getWidth(0) == width);
		BOOST_TEST(ptr->getHeight(0) == height);
		BOOST_TEST(ptr->getWidth(1) == width / 2);
		BOOST_TEST(ptr->getHeight(1) == height / 2);
		BOOST_TEST(ptr->getWidth(2) == width / 2);
		BOOST_TEST(ptr->getHeight(2) == height / 2);
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getStep(0) == step[0]);
		BOOST_TEST(ptr->getStep(1) == step[1]);
		BOOST_TEST(ptr->getStep(2) == step[2]);
		BOOST_TEST(ptr->getDataSize() == dataSize);
		for (auto i = 0; i < 3; i++)
		{
			BOOST_TEST(ptr->getNextPtrOffset(i) == nextPtrOffset[i]);
		}
	}

}

BOOST_AUTO_TEST_CASE(rawimageplanar_yuv444)
{
	int width = 1920;
	int height = 1080;
	size_t step[4] = { 2048, 2048, 2048 };
	size_t offset = static_cast<size_t>(2048 * height);
	size_t nextPtrOffset[4] = { 0, offset, offset * (2), 0 };
	int channels = 3;
	int depth = CV_8U;

	{
		auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV444, step, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		for (auto i = 0; i < channels; i++)
		{
			BOOST_TEST(ptr->getWidth(i) == width);
			BOOST_TEST(ptr->getHeight(i) == height);
			BOOST_TEST(ptr->getStep(i) == step[i]);
			BOOST_TEST(ptr->getNextPtrOffset(i) == nextPtrOffset[i]);
		}
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getDataSize() == step[0] * height * channels);
	}

	{
		auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV444, 512, depth, FrameMetadata::HOST));

		auto ptr = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		for (auto i = 0; i < channels; i++)
		{
			BOOST_TEST(ptr->getWidth(i) == width);
			BOOST_TEST(ptr->getHeight(i) == height);
			BOOST_TEST(ptr->getStep(i) == step[i]);
			BOOST_TEST(ptr->getNextPtrOffset(i) == nextPtrOffset[i]);
		}
		BOOST_TEST(ptr->getChannels() == channels);
		BOOST_TEST(ptr->getDataSize() == step[0] * height * channels);
	}
}

BOOST_AUTO_TEST_SUITE_END()