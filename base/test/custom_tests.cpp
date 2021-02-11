#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileWriterModule.h"
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
#include "EffectsNPPI.h"
#include "JPEGEncoderNVJPEG.h"
#include "JPEGDecoderNVJPEG.h"
#include "CudaStreamSynchronize.h"
#include "EffectsKernel.h"
# include "PipeLine.h"
#include <chrono>
BOOST_AUTO_TEST_SUITE(custom_tests)


BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    auto rawImagePin = fileReader->addOutputPin(metadata);
	
	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 1, height >> 1, stream)));
	copy->setNext(resize);
    
    EffectsNPPIProps effectsProps(stream);
	effectsProps.brightness = 0;
	effectsProps.contrast = 1;
	auto effects = boost::shared_ptr<EffectsNPPI>(new EffectsNPPI(effectsProps));
	resize->setNext(effects);

    auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
    effects->setNext(encoder);

	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/output.jpg")));
	
    // auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule("./data/output.jpg"));
    encoder->setNext(fileWriter);

	// auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	// m2->setNext(copy2);
	// auto outputPinId = fileWriter->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];


	// auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	// encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}


BOOST_AUTO_TEST_SUITE_END()
