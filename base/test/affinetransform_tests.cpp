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
	AffineTestsStruct(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth, int angle, int x, int y, float scale)
	{
		fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inpFilePath)));
		auto metadata = framemetadata_sp(new RawImageMetadata(width, height, imageType, bit_depth, 0, CV_8U, FrameMetadata::HOST, true));

		auto rawImagePin = fileReader->addOutputPin(metadata);

		auto stream = cudastream_sp(new ApraCudaStream);
		copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
		fileReader->setNext(copy1);

		AffineTransformProps affineProps(stream, angle, x, y, scale);
		m2 = boost::shared_ptr<Module>(new AffineTransform(affineProps));
		copy1->setNext(m2);
		copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
		m2->setNext(copy2);
		m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
		copy2->setNext(m3);

		BOOST_TEST(fileReader->init());
		BOOST_TEST(copy1->init());
		BOOST_TEST(m2->init());
		BOOST_TEST(copy2->init());
		BOOST_TEST(m3->init());
	}
	boost::shared_ptr<FileReaderModule>fileReader;
	boost::shared_ptr<Module>copy1;
	boost::shared_ptr<Module>m2;
	boost::shared_ptr<Module>copy2;
	boost::shared_ptr<ExternalSinkModule>m3;

	~AffineTestsStruct(){}	
};

BOOST_AUTO_TEST_CASE(MONO_rotation)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 5, 0, 0, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_scaling)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 0, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_shrink)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 0, 0.3f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_x)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 200, 0, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mnoo_shift_y)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 0, 100, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_scale)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, 100, 300, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_scale_rotate)
{
	AffineTestsStruct f("data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 5, 100, 300, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_rotation)
{
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 5, 0, 0, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_scaling)
{
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, 0, 0, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shifting)
{
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, 300, 100, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shift_scale_rotate)
{
	AffineTestsStruct f("data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 5, 300, 100, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_rotation)
{
	AffineTestsStruct f("data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 5, 0, 0, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_scaling)
{
	AffineTestsStruct f("data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 0, 0, 0, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_shifting)
{
	AffineTestsStruct f("data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 0, 300, 100, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_shift_scale_rotate)
{
	AffineTestsStruct f("data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 5, 300, 100, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}
	
BOOST_AUTO_TEST_CASE(RGBA_Image_rotation)
{
	AffineTestsStruct f("data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 5, 0, 0, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_Image_scaling)
{
	AffineTestsStruct f("data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, 0, 0, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_Image_shifting)
{
	AffineTestsStruct f("data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, 300, 100, 1.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_Image_shift_scale_rotate)
{
	AffineTestsStruct f("data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 5, 300, 100, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA)
{
	AffineTestsStruct f("data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::BGRA, CV_8UC4, 5, 300, 100, 3.0f);
	f.fileReader->step();
	f.copy1->step();
	f.m2->step();
	f.copy2->step();
	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv444)

{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("data/yuv444_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV444, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	AffineTransformProps affineProps(stream, 5, 100, 300, 2.0f);
	auto m2 = boost::shared_ptr<Module>(new AffineTransform(affineProps));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());

	fileReader->step();
	copy1->step();
	m2->step();
	copy2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_yuv444_1920x1080.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv420)

{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("data/yuv420_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	AffineTransformProps affineProps(stream, 5, 100, 300, 2.0f);
	auto m2 = boost::shared_ptr<Module>(new AffineTransform(affineProps));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());

	fileReader->step();
	copy1->step();
	m2->step();
	copy2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_yuv420_1920x1080.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
