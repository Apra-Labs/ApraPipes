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
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(resizenppi_tests)

BOOST_AUTO_TEST_CASE(mono_1920x1080)
{	
	auto width = 1920;
	auto height = 1080;
	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/resizenppi_tests_mono_1920x1080_to_960x540.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(overlay_1920x960_BGRA)
{
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);
	
	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/resizenppi_tests_overlay_1920x960_BGRA_to_960x480_C4.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);
	
	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
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

	Test_Utils::saveOrCompare("./data/testOutput/resizenppi_tests_yuv420_640x360_to_320x180.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(perf, *boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/4k.yuv")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto m2 = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
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
