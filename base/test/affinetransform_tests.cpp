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
#include "nppdefs.h"

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
	AffineTransformProps::Interpolation eInterpolation;

	AffineTestsStruct(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth,AffineTransformProps::Interpolation eInterpolation, int angle, int x, int y, float scale)
	{
		if (imageType == ImageMetadata::ImageType::YUV420 || imageType == ImageMetadata::ImageType::YUV444) {
			metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, imageType, size_t(0), CV_8U));
		}
		else {
			metadata = framemetadata_sp(new RawImageMetadata(width, height, imageType, bit_depth, 0, CV_8U, FrameMetadata::HOST, true));
		}
		createPipeline(inpFilePath, width, height,  imageType, bit_depth,  angle,  x,  y,  scale, eInterpolation);
	}
	void createPipeline(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth, int angle, int x, int y, float scale, AffineTransformProps::Interpolation eInterpolation)
	{
		fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inpFilePath)));	
		auto rawImagePin = fileReader->addOutputPin(metadata);

		auto stream = cudastream_sp(new ApraCudaStream);
		copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
		fileReader->setNext(copy1);

		AffineTransformProps affineProps(eInterpolation, stream, angle, x, y, scale);
		affineTransform = boost::shared_ptr<Module>(new AffineTransform(affineProps));
		copy1->setNext(affineTransform);
		copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
		affineTransform->setNext(copy2);
		sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
		copy2->setNext(sink);

		BOOST_TEST(fileReader->init());
		BOOST_TEST(copy1->init());
		BOOST_TEST(affineTransform->init());
		BOOST_TEST(copy2->init());
		BOOST_TEST(sink->init());
	}
	boost::shared_ptr<FileReaderModule>fileReader;
	boost::shared_ptr<Module>copy1;
	boost::shared_ptr<Module>affineTransform;
	boost::shared_ptr<Module>copy2;
	boost::shared_ptr<ExternalSinkModule>sink;

	~AffineTestsStruct()
	{
		sink->term();
		copy2->term();
		affineTransform->term();
		copy1->term();
		fileReader->term();
	}
};

BOOST_AUTO_TEST_CASE(MONO_rotation, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::NN, 5, 0, 0, 1.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_rotation_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_scale_rotate, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::NN, 45, 0, 0, 2.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_scale_rotate_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_shrink)
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::CUBIC, 0, 0, 0, 0.2);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-MONO_shrink_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_x, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::NN, 0, 100, 0, 1.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_x_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_y, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::NN, 0, 0, 300, 1.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_y_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_shift_scale_rotate)
{
	ImageMetadata::ImageType::MONO;
	AffineTestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, AffineTransformProps::CUBIC, 5, 100, 300, 2.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-mono_shift_scale_rotate_mono_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(RGB_Image_rotation, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, AffineTransformProps::NN, 90, 0, 0, 1.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_rotation_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
	
}

BOOST_AUTO_TEST_CASE(RGB_Image_scaling, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, AffineTransformProps::NN, 0, 0, 0, 2.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_scaling_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shifting,*boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, AffineTransformProps::NN, 0, 100, 300, 1.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_shifting_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_scale_rotate, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, AffineTransformProps::LINEAR, 30,0,0, 2.5);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_scale_rotate_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::RGB;
	AffineTestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, AffineTransformProps::LINEAR, 5, 100, 300, 2.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests-RGB_Image_shift_scale_rotate_frame_1280x720_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_Image_shift_scale_rotate,*boost::unit_test::disabled())
{
	ImageMetadata::ImageType::BGR;
	AffineTestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, AffineTransformProps::NN, 5, 300, 100, 3.0);
	
	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_BGR_1080x720.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(RGBA_Image_scale_rotate, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::RGBA;
	AffineTestsStruct f("./data/rgba_400x400.raw", 400,400, ImageMetadata::ImageType::RGBA, CV_8UC4, AffineTransformProps::NN,90, 0, 0, 2.5);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_400x400_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_Image_shift_scale_rotate)
{
	ImageMetadata::ImageType::RGBA;
	AffineTestsStruct f("./data/rgba_400x400.raw", 400, 400, ImageMetadata::ImageType::RGBA, CV_8UC4, AffineTransformProps::NN, 40, 100, 200, 2.5);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_frame_all_400x400_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}
	
BOOST_AUTO_TEST_CASE(BGRA_Image_shift_scale_rotate, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::BGRA;
	AffineTestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::BGRA, CV_8UC4, AffineTransformProps::LINEAR, 90, 100,100, 2.5);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_8bit_frame_1280x720_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV444_scale_rotate, *boost::unit_test::disabled())
{
	ImageMetadata::ImageType::YUV444;
	AffineTestsStruct f("./data/yuv444.raw", 400, 400, ImageMetadata::ImageType::YUV444, size_t(0), AffineTransformProps::NN, 90, 0, 0, 2.5);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_YUV444_scale_rotate_yuv444.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}
BOOST_AUTO_TEST_CASE(YUV444_shift_scale_rotate)
{
	ImageMetadata::ImageType::YUV444;
	AffineTestsStruct f("./data/yuv444_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::YUV444, size_t(0), AffineTransformProps::CUBIC2P_CATMULLROM,5, 200, 300, 2.0);

	f.fileReader->step();
	f.copy1->step();
	f.affineTransform->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_tests_yuv444_1920x1080.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(Host_Mono, *boost::unit_test::disabled())
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto affine = boost::shared_ptr<Module>(new AffineTransform(AffineTransformProps(-45,0,0,2.0)));
    fileReader->setNext(affine);

	auto outputPinId = affine->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	affine->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(affine->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	affine->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_host.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(Host_RGB)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto affine = boost::shared_ptr<Module>(new AffineTransform(AffineTransformProps(-30, 10, 10, 2.0)));
	fileReader->setNext(affine);

	auto outputPinId = affine->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	affine->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(affine->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	affine->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/affinetransform_host_RGB.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
