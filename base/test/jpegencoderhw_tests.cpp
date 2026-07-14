#include <boost/test/unit_test.hpp>
#include "JPEGEncoderL4TM.h"
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "MemTypeConversion.h"
#include "ExternalSinkModule.h"
#include "Logger.h"
#include "PipeLine.h"
#include "RawImageMetadata.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(jpegencoderhw_tests)

// Test: FileReader -> MemTypeConversion(HOST->DMABUF) -> JPEGEncoderL4TM(HW) -> FileWriter
BOOST_AUTO_TEST_CASE(hw_jpeg_encode_rgba_1280x720_dmabuf)
{
#ifdef ARM64
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(logprops);

	// FileReader with RGBA image
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(
		FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, 
		ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	// MemTypeConversion: HOST -> DMABUF
	auto memConversion = boost::shared_ptr<Module>(new MemTypeConversion(
		MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(memConversion);

	// JPEGEncoderL4TM with hardware acceleration enabled
	JPEGEncoderL4TMProps encoderProps;
	encoderProps.quality = 90;
	encoderProps.useHardwareAcceleration = true;
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	memConversion->setNext(jpegEncoder);

	// Sink to capture output
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	jpegEncoder->setNext(sink);

	// Initialize pipeline
	BOOST_TEST(fileReader->init());
	BOOST_TEST(memConversion->init());
	BOOST_TEST(jpegEncoder->init());
	BOOST_TEST(sink->init());

	// Execute pipeline
	fileReader->step();
	memConversion->step();
	jpegEncoder->step();

	// Verify output
	auto frames = sink->pop();
	BOOST_TEST(!frames.empty());
	
	auto outputPinId = jpegEncoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(outFrame->size() > 0);

	// Save output JPEG
	Test_Utils::saveOrCompare("./data/testOutput/hw_jpeg_rgba_1280x720.jpg", 
		(const uint8_t *)outFrame->data(), outFrame->size(), 0);
	
	LOG_INFO << "Hardware JPEG encoder test RGBA passed. Output size: " << outFrame->size() << " bytes";
#endif
}

// Test: FileReader -> MemTypeConversion(HOST->DMABUF) -> JPEGEncoderL4TM(HW) -> FileWriter (NV12)
BOOST_AUTO_TEST_CASE(hw_jpeg_encode_nv12_1280x720_dmabuf)
{
#ifdef ARM64
	// FileReader with NV12 image
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(
		FileReaderModuleProps("./data/nv12_1280x720.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, 
		ImageMetadata::ImageType::NV12, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	// MemTypeConversion: HOST -> DMABUF
	auto memConversion = boost::shared_ptr<Module>(new MemTypeConversion(
		MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(memConversion);

	// JPEGEncoderL4TM with hardware acceleration
	JPEGEncoderL4TMProps encoderProps;
	encoderProps.quality = 85;
	encoderProps.useHardwareAcceleration = true;
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	memConversion->setNext(jpegEncoder);

	// Sink
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	jpegEncoder->setNext(sink);

	// Initialize
	BOOST_TEST(fileReader->init());
	BOOST_TEST(memConversion->init());
	BOOST_TEST(jpegEncoder->init());
	BOOST_TEST(sink->init());

	// Execute
	fileReader->step();
	memConversion->step();
	jpegEncoder->step();

	// Verify
	auto frames = sink->pop();
	BOOST_TEST(!frames.empty());
	
	auto outputPinId = jpegEncoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(outFrame->size() > 0);

	// Save output
	Test_Utils::saveOrCompare("./data/testOutput/hw_jpeg_nv12_1280x720.jpg", 
		(const uint8_t *)outFrame->data(), outFrame->size(), 0);
	
	LOG_INFO << "Hardware JPEG encoder test NV12 passed. Output size: " << outFrame->size() << " bytes";
#endif
}

// Test: Software fallback when hardware acceleration is disabled
BOOST_AUTO_TEST_CASE(sw_jpeg_encode_fallback_rgba_1280x720)
{
	// FileReader with RGBA image
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(
		FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, 
		ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	// JPEGEncoderL4TM with hardware acceleration DISABLED
	JPEGEncoderL4TMProps encoderProps;
	encoderProps.quality = 90;
	encoderProps.useHardwareAcceleration = false;  // Force software
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	fileReader->setNext(jpegEncoder);

	// Sink
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	jpegEncoder->setNext(sink);

	// Initialize
	BOOST_TEST(fileReader->init());
	BOOST_TEST(jpegEncoder->init());
	BOOST_TEST(sink->init());

	// Execute
	fileReader->step();
	jpegEncoder->step();

	// Verify
	auto frames = sink->pop();
	BOOST_TEST(!frames.empty());
	
	auto outputPinId = jpegEncoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(outFrame->size() > 0);

	// Save output
	Test_Utils::saveOrCompare("./data/testOutput/sw_jpeg_rgba_1280x720.jpg", 
		(const uint8_t *)outFrame->data(), outFrame->size(), 0);
	
	LOG_INFO << "Software JPEG encoder fallback test passed. Output size: " << outFrame->size() << " bytes";
}

// Test: Pipeline test with FileWriter
BOOST_AUTO_TEST_CASE(hw_jpeg_pipeline_rgba_filewriter, * boost::unit_test::disabled())
{
#ifdef ARM64
	// Create pipeline
	PipeLine p("hw_jpeg_test_pipeline");

	// FileReader
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(
		FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, 
		ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	// MemTypeConversion
	auto memConversion = boost::shared_ptr<Module>(new MemTypeConversion(
		MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(memConversion);

	// JPEG Encoder
	JPEGEncoderL4TMProps encoderProps;
	encoderProps.quality = 95;
	encoderProps.useHardwareAcceleration = true;
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	memConversion->setNext(jpegEncoder);

	// FileWriter
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(
		FileWriterModuleProps("./data/testOutput/hw_jpeg_pipeline_output.jpg")));
	jpegEncoder->setNext(fileWriter);

	// Add to pipeline
	p.appendModule(fileReader);

	// Run pipeline
	BOOST_TEST(p.init());
	p.run_all_threaded();
	p.step();
	p.stop();
	p.term();
	p.wait_for_all();

	LOG_INFO << "Hardware JPEG pipeline test completed";
#endif
}

BOOST_AUTO_TEST_SUITE_END()
