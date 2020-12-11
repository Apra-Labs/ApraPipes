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
#include "JPEGEncoderNVJPEG.h"
#include "JPEGDecoderNVJPEG.h"
#include "ResizeNPPI.h"
#include "FileWriterModule.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(nvjpeg_combo_tests)

BOOST_AUTO_TEST_CASE(decode_encode_mono_1920x960)
{	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	auto encoder = boost::shared_ptr<Module>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	decoder->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());	
	
	fileReader->step();
	decoder->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/nvjpeg_combo_tests_decode_encode_mono_1920x960.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(decode_resize_encode_mono_1920x960)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
	decoder->setNext(resize);

	auto encoder = boost::shared_ptr<Module>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	resize->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	decoder->step();
	resize->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/nvjpeg_combo_tests_decode_resize_encode_mono_1920x960_to_960x480.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(decode_resize_cc_raw_1920x960, *boost::unit_test::disabled())
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/testOutput/effects/hue_img_864x576")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(1920, 960, stream)));
	decoder->setNext(resize);	
	
	auto cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
	resize->setNext(cc);

	auto copy1 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	cc->setNext(copy1);

	Test_Utils::createDirIfNotExist("./data/testOutput/effectsraw/frame_00.raw");
	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/effectsraw/frame_??.raw")));
	copy1->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(cc->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(sink->init());
	
	fileReader->play(true);

	for (auto i = 0; i < 27; i++)
	{
		fileReader->step();
		decoder->step();
		resize->step();
		cc->step();
		copy1->step();
		sink->step();
	}

}

BOOST_AUTO_TEST_CASE(decode_encode_color_yuv420_640x360)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/color_yuv420_640x360.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	auto encoder = boost::shared_ptr<Module>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	decoder->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	decoder->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/nvjpeg_combo_tests_decode_encode_color_yuv420_640x360.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(decode_resize_encode_color_yuv420_640x360)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/color_yuv420_640x360.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);

	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
	decoder->setNext(resize);

	auto encoder = boost::shared_ptr<Module>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	resize->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	decoder->step();
	resize->step();
	encoder->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/nvjpeg_combo_tests_decode_resize_encode_color_yuv420_640x360_to_960x480.jpg", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
