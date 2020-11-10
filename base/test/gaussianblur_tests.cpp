#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "GaussianBlur.h"

#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(gaussianblur_tests)

BOOST_AUTO_TEST_CASE(mono_1920x1080)
{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream->getCudaStream())));
	fileReader->setNext(copy1);

	GaussianBlurProps props(stream, 11);
	auto blur = boost::shared_ptr<GaussianBlur>(new GaussianBlur(props));
	copy1->setNext(blur);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream->getCudaStream())));
	blur->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(blur->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	{
		fileReader->step();
		copy1->step();
		blur->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/mono_1920x1080_blur_11.raw";
		Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	props.kernelSize = 3;
	blur->setProps(props);
	blur->step();

	{
		fileReader->step();
		copy1->step();
		blur->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/mono_1920x1080_blur_3.raw";
		Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_SUITE_END()
