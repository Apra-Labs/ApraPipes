#include <boost/test/unit_test.hpp>
#include "AffineTransform.h"
#include "FileReaderModule.h"
#include "RawImageMetadata.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "CudaMemCopy.h"
#include "ExternalSinkModule.h"
#include "AIPExceptions.h"
#include "stdafx.h"
#include "PipeLine.h"


BOOST_AUTO_TEST_SUITE(affinetransform_tests)

struct AffineTestsStruct
{
	std::string inpFilePath;
	int width;
	int height;
	ImageMetadata::ImageType imageType;
	int bit_depth;
	int angle;
	int x;
	int y;
	float scale;
	framemetadata_sp metadata;

	AffineTestsStruct::AffineTestsStruct(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth, int angle, int x, int y, float scale)
		: inpFilePath(inpFilePath), width(width), height(height), imageType(imageType), bit_depth(bit_depth), angle(angle), x(x), y(y), scale(scale)
	{
		if (imageType == ImageMetadata::ImageType::YUV420 || imageType == ImageMetadata::ImageType::YUV444) {
			metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, imageType, size_t(0), CV_8U));
		}
		else {
			metadata = framemetadata_sp(new RawImageMetadata(width, height, imageType, bit_depth, 0, CV_8U, FrameMetadata::HOST, true));
		}
		createPipeline();
	}
	void createPipeline()
	{
		fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inpFilePath)));	
		auto rawImagePin = fileReader->addOutputPin(metadata);

		auto stream = cudastream_sp(new ApraCudaStream);
		copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
		fileReader->setNext(copy1);

		AffineTransformProps affineProps(stream, angle, x, y, scale);
		affineTransform = boost::shared_ptr<Module>(new AffineTransform(affineProps));
		copy1->setNext(affineTransform);
		copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
		affineTransform->setNext(copy2);
		m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
		copy2->setNext(m3);

		BOOST_TEST(fileReader->init());
		BOOST_TEST(copy1->init());
		BOOST_TEST(affineTransform->init());
		BOOST_TEST(copy2->init());
		BOOST_TEST(m3->init());
	}
	boost::shared_ptr<FileReaderModule>fileReader;
	boost::shared_ptr<Module>copy1;
	boost::shared_ptr<Module>affineTransform;
	boost::shared_ptr<Module>copy2;
	boost::shared_ptr<ExternalSinkModule>m3;
	~AffineTestsStruct(){}	
};

BOOST_AUTO_TEST_CASE(MONO_rotation, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 5, 0, 0, 1.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_rotation_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_scale, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 0, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_scale_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}
BOOST_AUTO_TEST_CASE(MONO_shrink)
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 0, 0.2f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_shrink_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_x, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 100, 0, 1.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_x_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_y, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 300, 1.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_y_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_scale_rotate)
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 5, 100, 300, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_scale_rotate_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(RGB_Image_rotation, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 5, 0, 0, 1.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_rotation_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
	
}

BOOST_AUTO_TEST_CASE(RGB_Image_scaling, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, 0, 0, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_scaling_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shifting,*boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, 100, 300, 1.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_shifting_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 5, 100, 300, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_shift_scale_rotate_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::BGR;
	AffineTestsStruct f("data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 5, 300, 100, 3.0f);
	
	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);

}
	
BOOST_AUTO_TEST_CASE(RGBA_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::RGBA;
	AffineTestsStruct f("data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 5, 300, 100, 3.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::BGRA;
	AffineTestsStruct f("data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::BGRA, CV_8UC4, 5, 300, 100, 3.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV444_shift_scale_rotate)
{
	ImageMetadata::ImageType::YUV444;
	AffineTestsStruct f("data/yuv444_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::YUV444, size_t(0),5, 200, 300, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_yuv444_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_shift_scale_rotate)
{
	ImageMetadata::ImageType::YUV420;
	AffineTestsStruct f("data/yuv420_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::YUV420, size_t(0), 5, 200, 300, 2.0f);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_yuv420_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
