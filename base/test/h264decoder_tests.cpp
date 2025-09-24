#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "H264Decoder.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "H264Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#include "MemTypeConversion.h"

#include "CudaMemCopy.h"
#include "nv_test_utils.h"
#include "ResizeNPPI.h"
#include "JPEGEncoderL4TM.h"

BOOST_AUTO_TEST_SUITE(h264decoder_tests)

BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer_2)
{
	Logger::setLogLevel("debug");

        auto fileReader =
            boost::shared_ptr<FileReaderModule>(new FileReaderModule(
                FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
        auto metadata = framemetadata_sp(
            new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGBA,
                                 CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
        fileReader->addOutputPin(metadata);

        auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversionDMA = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	fileReader->setNext(memconversionDMA);

	auto memconversionHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversionDMA->setNext(memconversionHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MEMCPY_TEST/frame_????.raw")));
	memconversionHost->setNext(fileWriter);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

        if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(1);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}


BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer)
{
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	Decoder->setNext(memconversion2);

	
	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/yash_frame_????.raw")));
	memconversion2->setNext(fileWriter);


	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(15);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}


BOOST_AUTO_TEST_CASE(sample_mp4_file_decoder_cuda_device_to_host)
{
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	Decoder->setNext(memconversion2);

	auto hostToDevice = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, stream)));
	memconversion2->setNext(hostToDevice);


	auto resizeNPPI = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(320, 180, stream)));
	hostToDevice->setNext(resizeNPPI);


	auto deviceToHost = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyDeviceToHost, stream)));
	resizeNPPI->setNext(deviceToHost);
	
	auto jpegEncoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = jpegEncoder->addOutputPin(encodedImageMetadata);
	deviceToHost->setNext(jpegEncoder);
	
	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/ULT_TEST/frame_????.jpg")));
	jpegEncoder->setNext(fileWriter);


	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(15);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}


BOOST_AUTO_TEST_SUITE_END()