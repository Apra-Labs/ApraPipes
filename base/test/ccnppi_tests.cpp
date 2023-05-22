#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "ResizeNPPI.h"
#include "CCNPPI.h"
#include "JPEGEncoderNVJPEG.h"
#include "test_utils.h"
#include "nv_test_utils.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(ccnppi_tests)

struct CCNPPITestsStruct
{
	std::string inpFilePath;
	int width;
	int height;
	ImageMetadata::ImageType imageType;
	int bit_depth;
	framemetadata_sp metadata;

	CCNPPITestsStruct(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth, ImageMetadata::ImageType OimageType)
	{
		if (imageType == ImageMetadata::ImageType::YUV420 || imageType == ImageMetadata::ImageType::NV12) {
			metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, imageType, size_t(0), CV_8U));
		}
		else {
			metadata = framemetadata_sp(new RawImageMetadata(width, height, imageType, bit_depth, 0, CV_8U, FrameMetadata::HOST, true));
		}

		createPipeline(inpFilePath, width, height, imageType, bit_depth, OimageType);
	}

	void createPipeline(const std::string& inpFilePath, int width, int height, ImageMetadata::ImageType imageType, int bit_depth, ImageMetadata::ImageType OimageType)
	{
		fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inpFilePath)));
		auto rawImagePin = fileReader->addOutputPin(metadata);

		auto stream = cudastream_sp(new ApraCudaStream);
		copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
		fileReader->setNext(copy1);

		CCNPPIProps CCProps(OimageType,stream);
		ccnppi = boost::shared_ptr<Module>(new CCNPPI(CCProps));
		copy1->setNext(ccnppi);

		copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
		ccnppi->setNext(copy2);

		m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
		copy2->setNext(m3);

		PipeLine p("test");
		p.appendModule(fileReader);

		BOOST_TEST(fileReader->init());
		BOOST_TEST(copy1->init());
		BOOST_TEST(ccnppi->init());
		BOOST_TEST(copy2->init());
		BOOST_TEST(m3->init());
	}

	boost::shared_ptr<FileReaderModule>fileReader;
	boost::shared_ptr<Module>copy1;
	boost::shared_ptr<Module>ccnppi;
	boost::shared_ptr<Module>copy2;
	boost::shared_ptr<ExternalSinkModule>m3;

	~CCNPPITestsStruct()
	{
		m3->term();
		copy2->term();
		ccnppi->term();
		copy1->term();
		fileReader->term();
	}
};

BOOST_AUTO_TEST_CASE(MONO_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::MONO;
	CCNPPITestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/mono_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::MONO;
	CCNPPITestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/mono_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::MONO;
	CCNPPITestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/mono_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::MONO;
	CCNPPITestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/mono_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(MONO_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::MONO;
	CCNPPITestsStruct f("./data/mono_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::MONO, CV_8UC1, ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/mono_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGB;
	CCNPPITestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgb_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGB;
	CCNPPITestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgb_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGB;
	CCNPPITestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgb_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGB;
	CCNPPITestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgb_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGB_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGB;
	CCNPPITestsStruct f("./data/frame_1280x720_rgb.raw", 1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/rgb_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGR;
	CCNPPITestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgr_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGR;
	CCNPPITestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgr_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGR;
	CCNPPITestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgr_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGR;
	CCNPPITestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgr_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGR_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGR;
	CCNPPITestsStruct f("./data/BGR_1080x720.raw", 1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/bgr_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGBA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgba_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGBA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgba_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGBA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgba_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGBA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/rgba_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(RGBA_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::RGBA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_rgba.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/rgba_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGRA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgra_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGRA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgra_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGRA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgra_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGRA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/bgra_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(BGRA_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::BGRA;
	CCNPPITestsStruct f("./data/8bit_frame_1280x720_bgra.raw", 1280, 720, ImageMetadata::ImageType::BGRA, CV_8UC4, ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/bgra_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/yuv420_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/YUV420_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/yuv420_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/yuv420_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/yuv420_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_MONO,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704,576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::MONO);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_mono.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_RGB,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704, 576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::RGB);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_rgb.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_BGR,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704, 576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::BGR);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_bgr.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_RGBA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704, 576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::RGBA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_rgba.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_BGRA,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704, 576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::BGRA);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_bgra.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(NV12_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::NV12;
	CCNPPITestsStruct f("./data/nv12-704x576.raw", 704, 576, ImageMetadata::ImageType::NV12, size_t(0), ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/nv12_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv411_I_1920x1080,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV411_I;
	CCNPPITestsStruct f("./yuv411_I_1920x1080.raw", 1920, 1080, ImageMetadata::ImageType::YUV411_I, CV_8UC3, ImageMetadata::ImageType::YUV444);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/yuv411_to_yuv444.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(YUV420_to_YUV420,
	*utf::precondition(if_compute_cap_supported()))
{
	ImageMetadata::ImageType::YUV420;
	CCNPPITestsStruct f("./data/yuv420_640x360.raw", 640, 360, ImageMetadata::ImageType::YUV420, size_t(0), ImageMetadata::ImageType::YUV420);

	f.fileReader->step();
	f.copy1->step();
	f.ccnppi->step();
	f.copy2->step();

	auto outputPinId = f.copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = f.m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/yuv420_to_yuv420.raw", (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(perf, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	auto width = 1920;
	auto height = 1080;
	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv411_I_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::YUV411_I, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV444, stream)));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];


	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	PipeLine p("test");
	p.appendModule(fileReader);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());

	for (auto i = 0; i < 1; i++)
	{
		fileReader->step();
		copy1->step();
		m2->step();
		copy2->step();
		m3->pop();
	}
	p.stop();
	p.term();

}

BOOST_AUTO_TEST_SUITE_END()
