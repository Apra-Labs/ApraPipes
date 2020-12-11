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

BOOST_AUTO_TEST_SUITE(ccnppi_tests)

BOOST_AUTO_TEST_CASE(yuv411_I_1920x1080)
{	
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

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_yuv411_I_1920x1080_to_yuv444_1920x1080.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_1920x1080_to_bgra_1920x1080)
{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];


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
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_mono_1920x1080_to_bgra_1920x1080.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(overlay_1920x960_BGRA_to_yuv420_1920x960)
{
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_overlay_1920x960_BGRA_to_yuv420_1920x960.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_to_bgra_640x360)
{
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
	copy1->setNext(m2);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	m2->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];


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
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_yuv420_640x360_to_bgra_640x360.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv411_I_1920x1080__resize_to_jpg)
{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv411_I_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::YUV411_I, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	auto cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV444, stream)));
	copy->setNext(cc);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
	cc->setNext(resize);
	
	auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	resize->setNext(encoder);
	auto outputPinId = encoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];


	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(cc->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());



	fileReader->step();
	copy->step();
	cc->step();
	resize->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_yuv411_I_1920x1080_to_yuv444_960x540.jpg", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_1920x1080_to_yuv420_1920x1080)
{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_mono_1920x1080_to_yuv420_1920x1080.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_1920x960_to_yuv420_1920x960)
{
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_mono_1920x960_to_yuv420_1920x960.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_to_yuv420_640x360)
{
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/ccnppi_tests_yuv420_640x360_to_yuv420_640x360.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
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
}

BOOST_AUTO_TEST_SUITE_END()
