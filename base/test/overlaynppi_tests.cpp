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
#include "CCNPPI.h"
#include "ResizeNPPI.h"
#include "OverlayNPPI.h"
#include "JPEGEncoderNVJPEG.h"
#include "test_utils.h"
#include "MetadataHints.h"

BOOST_AUTO_TEST_SUITE(overlaynppi_tests)

BOOST_AUTO_TEST_CASE(mono_1920x1080)
{	
	auto stream = cudastream_sp(new ApraCudaStream);

	auto width = 1920;
	auto height = 1080;
	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto source_cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
	copy1->setNext(source_cc);

	auto overlay_host = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto overlay_metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));	
	overlay_metadata->setHint(OVERLAY_HINT);
	overlay_host->addOutputPin(overlay_metadata);

	auto overlay_copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	overlay_host->setNext(overlay_copy);	

	auto overlay = boost::shared_ptr<Module>(new OverlayNPPI(OverlayNPPIProps(stream)));
	overlay_copy->setNext(overlay);
	source_cc->setNext(overlay);
	
	auto output_cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
	overlay->setNext(output_cc);
	auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	output_cc->setNext(encoder);
	auto outputPinId = encoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];


	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(source_cc->init());
	BOOST_TEST(overlay_host->init());
	BOOST_TEST(overlay_copy->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(output_cc->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());	
	
	overlay_host->step();
	overlay_copy->step();
	overlay->step();
	fileReader->step();
	copy1->step();
	source_cc->step();
	overlay->step();
	output_cc->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/overlaynppi_tests_mono_1920x1080_to_overlay_1920x1080_bgra.jpg", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(mono_1920x1080_pos)
{
	auto stream = cudastream_sp(new ApraCudaStream);

	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto source_cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
	copy1->setNext(source_cc);

	auto overlay_host = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto overlay_metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	overlay_metadata->setHint(OVERLAY_HINT);
	overlay_host->addOutputPin(overlay_metadata);

	auto overlay_copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	overlay_host->setNext(overlay_copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
	overlay_copy->setNext(resize);

	auto overlayProps = OverlayNPPIProps(stream);
	overlayProps.offsetX = width >> 1;
	overlayProps.offsetY = height >> 1;
	auto overlay = boost::shared_ptr<OverlayNPPI>(new OverlayNPPI(overlayProps));
	resize->setNext(overlay);
	source_cc->setNext(overlay);

	auto output_cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::YUV420, stream)));
	overlay->setNext(output_cc);
	auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	output_cc->setNext(encoder);
	auto outputPinId = encoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];


	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(source_cc->init());
	BOOST_TEST(overlay_host->init());
	BOOST_TEST(overlay_copy->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(output_cc->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());


	for (auto i = 0; i < 5; i++)
	{
		overlay_host->step();
		overlay_copy->step();
		resize->step();
		overlay->step();
		fileReader->step();
		copy1->step();
		source_cc->step();
		overlay->step();
		output_cc->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		Test_Utils::saveOrCompare(std::string("./data/testOutput/overlaynppi_tests_mono_1920x1080_pos_" + std::to_string(overlayProps.offsetX) + "_" + std::to_string(overlayProps.offsetY) + "_to_overlay_1920x1080_bgra.jpg").c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);

		overlayProps.offsetX  -= 10;
		overlayProps.offsetY -= 10;
		overlay->setProps(overlayProps);
		overlay->step();
	}

	overlayProps.offsetX = width >> 1;
	overlayProps.offsetY = height >> 1;
	overlayProps.globalAlpha = 80;
	overlay->setProps(overlayProps);
	overlay->step();

	for (auto i = 0; i < 5; i++)
	{
		overlay_host->step();
		overlay_copy->step();
		resize->step();
		overlay->step();
		fileReader->step();
		copy1->step();
		source_cc->step();
		overlay->step();
		output_cc->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		Test_Utils::saveOrCompare(std::string("./data/testOutput/overlaynppi_tests_mono_1920x1080_pos_" + std::to_string(overlayProps.offsetX) + "_" + std::to_string(overlayProps.offsetY) + "_to_overlay_1920x1080_bgra_globalalpha.jpg").c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);

		overlayProps.offsetX -= 10;
		overlayProps.offsetY -= 10;
		overlay->setProps(overlayProps);
		overlay->step();
	}
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

	auto overlay_host = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_640x360_yuv420.raw")));
	auto overlay_metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	overlay_metadata->setHint(OVERLAY_HINT);
	overlay_host->addOutputPin(overlay_metadata);

	auto overlay_copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	overlay_host->setNext(overlay_copy);
	
	auto overlayProps = OverlayNPPIProps(stream);
	auto overlay = boost::shared_ptr<OverlayNPPI>(new OverlayNPPI(overlayProps));
	overlay_copy->setNext(overlay);
	copy1->setNext(overlay);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	overlay->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];


	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(overlay_host->init());
	BOOST_TEST(overlay_copy->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());


	overlay_host->step();
	overlay_copy->step();
	overlay->step();

	{
		fileReader->step();
		copy1->step();
		overlay->step();
		copy2->step();
		auto frames = m3->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

		Test_Utils::saveOrCompare("./data/testOutput/overlaynppi_tests_yuv420_640x360_to_overlay_640x360_yuv420.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	{
		overlayProps.globalAlpha = 200;
		overlay->setProps(overlayProps);
		overlay->step();

		fileReader->step();
		copy1->step();
		overlay->step();
		copy2->step();
		auto frames = m3->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

		Test_Utils::saveOrCompare("./data/testOutput/overlaynppi_tests_yuv420_640x360_to_overlay_640x360_yuv420_globalalpha.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_pos)
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
			
	auto overlay_host = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_640x360_yuv420.raw")));
	auto overlay_metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	overlay_metadata->setHint(OVERLAY_HINT);
	overlay_host->addOutputPin(overlay_metadata);

	auto overlay_copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	overlay_host->setNext(overlay_copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(320, 180, stream)));
	overlay_copy->setNext(resize);

	auto overlayProps = OverlayNPPIProps(stream);
	overlayProps.offsetX = width >> 1;
	overlayProps.offsetY = height >> 1;
	auto overlay = boost::shared_ptr<Module>(new OverlayNPPI(overlayProps));
	resize->setNext(overlay);
	copy1->setNext(overlay);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	overlay->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];


	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(m3);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(overlay_host->init());
	BOOST_TEST(overlay_copy->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(m3->init());


	overlay_host->step();
	overlay_copy->step();
	resize->step();
	overlay->step();

	fileReader->step();
	copy1->step();
	overlay->step();
	copy2->step();
	auto frames = m3->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

	Test_Utils::saveOrCompare("./data/testOutput/overlaynppi_tests_yuv420_640x360_pos_to_overlay_640x360_yuv420.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
