#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "H264Decoder.h"
#include "test_utils.h"
#include "DMAFDToHostCopy.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "H264Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#include "MemTypeConversion.h"

#include "CudaMemCopy.h"
#include "nv_test_utils.h"


BOOST_AUTO_TEST_SUITE(h264decoder_tests)

BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer)
{
	Logger::setLogLevel("info");
	auto stream = cudastream_sp(new ApraCudaStream);
	// metadata is known
	std::string videoPath = "/home/developer/workspace/ApraPipes/1684824632.mp4";
	// std::string videoPath = "/media/developer/7C3B-7A0B/2023-07-11/DOCTOR/PATIENT/2023-07-11_16-14-50-880.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	// StatSinkProps sinkProps;
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink2 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	// mp4Reader->setNext(sink2);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
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


BOOST_AUTO_TEST_SUITE_END()