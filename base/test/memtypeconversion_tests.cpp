#include <boost/test/unit_test.hpp>
#include "MemTypeConversion.h"
#include "FileReaderModule.h"
#include "Logger.h"
#include "PipeLine.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FileWriterModule.h"
#include "test_utils.h"
#include "ExternalSinkModule.h"
#include "ImageDecoderCV.h"
#include "FileWriterModule.h"
#include "JPEGDecoderL4TM.h"
#if defined(__arm__) || defined(__aarch64__)
#include "NvTransform.h"
#include "EglRenderer.h"
#endif

BOOST_AUTO_TEST_SUITE(memtypeconversion_tests)

BOOST_AUTO_TEST_CASE(Device_to_Dma_RGBA, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
    FileReaderModuleProps fileReaderProps("/media/developer/7979-7B01/2024-01-24/D1/P1/2024-01-24_16-37-49-137.jpeg");
    fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto decoder = boost::shared_ptr<JPEGDecoderL4TM>(new JPEGDecoderL4TM());
	fileReader->setNext(decoder);
	auto rawImageMetadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = decoder->addOutputPin(rawImageMetadata);


    // auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/reader/frame_????.raw")));
	// decoder->setNext(fileWriter);


	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	decoder->setNext(copy1);

	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	copy1->setNext(memconversion);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0, 0)));
	memconversion->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(20));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_SUITE_END()
