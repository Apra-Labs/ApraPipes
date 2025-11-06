#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "MemTypeConversion.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "JPEGEncoderL4TM.h"
#include "FileReaderModule.h"
#include "StatSink.h"
#include "PipeLine.h"
#include "test_utils.h"
#include <fstream>
#include "NvV4L2Camera.h"
#include "DMAFDToHostCopy.h"
#include "FileWriterModule.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#include "nvbufsurface.h"

#include "NvArgusCamera.h"

BOOST_AUTO_TEST_SUITE(jpegencoderl4tm_tests)

BOOST_AUTO_TEST_CASE(argus_jpeg, * boost::unit_test::disabled())
{
	LoggerProps logProps;
    logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
	NvArgusCameraProps sourceProps(1280, 720);
	sourceProps.maxConcurrentFrames = 10;
	sourceProps.fps = 30;
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/nv12-argus.raw", true)));
	copySource->setNext(fileWriter);
	

	auto jencode = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = jencode->addOutputPin(encodedImageMetadata);
	source->setNext(jencode);


	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvargus/frame_argus.jpg", true)));
	jencode->setNext(fileWriter2);
    PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p.stop();
	p.term();

	p.wait_for_all();
}



BOOST_AUTO_TEST_CASE(jpegencoder_dmabuf_fd_nv12, * boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);

    const int width = 704, height = 576;
    const std::string inputYuvPath = "./data/nv12-704x576.raw";

    // Source
    auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());

    // Create DMABUF metadata with actual pitches from NvBufSurface
    auto meta = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::DMABUF));
    DMAAllocator::setMetadata(meta, width, height, ImageMetadata::NV12);
    auto rawPin = m1->addOutputPin(meta);

    // NVJPEG encoder module
    auto enc = boost::shared_ptr<Module>(new JPEGEncoderL4TM()); // ensure this module exists/linked
    m1->setNext(enc);
    auto encMeta = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    auto encPin = enc->addOutputPin(encMeta);

    // Sink
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    enc->setNext(sink);

    BOOST_TEST(m1->init());
    BOOST_TEST(enc->init());
    BOOST_TEST(sink->init());

    // Allocate DMABUF frame and fill with I420 data (row-wise using NvBufSurface params)
    auto frame = m1->makeFrame(meta->getDataSize(), rawPin);
    {
        auto dma = static_cast<DMAFDWrapper*>(frame->data());
        NvBufSurface* surf = dma->getNvBufSurface();

        std::ifstream in(inputYuvPath, std::ios::binary);
        BOOST_TEST(in.good());

        // planes: 0=Y,1=U,2=V
        for (int p = 0; p < 3; ++p) {
            NvBufSurfaceMap(surf, 0, p, NVBUF_MAP_READ_WRITE);
            NvBufSurfaceSyncForCpu(surf, 0, p);
            auto dst   = static_cast<uint8_t*>(surf->surfaceList[0].mappedAddr.addr[p]);
            auto rows  = surf->surfaceList[0].planeParams.height[p];
            auto pitch = surf->surfaceList[0].planeParams.pitch[p];
            auto rowB  = surf->surfaceList[0].planeParams.width[p] *
                         surf->surfaceList[0].planeParams.bytesPerPix[p];

            std::vector<uint8_t> row(rowB);
            for (uint32_t r = 0; r < rows; ++r) {
                in.read(reinterpret_cast<char*>(row.data()), rowB);
                BOOST_TEST(static_cast<size_t>(in.gcount()) == rowB);
                memcpy(dst + r * pitch, row.data(), rowB);
            }
            NvBufSurfaceSyncForDevice(surf, 0, p);
            NvBufSurfaceUnMap(surf, 0, p);
        }
    }

    frame_container fc; fc.insert({ rawPin, frame });
    m1->send(fc);
    enc->step();

    auto out = sink->pop();
    BOOST_TEST((out.find(encPin) != out.end()));
    auto outFrame = out[encPin];
    BOOST_TEST(outFrame->size() > 0);

    Test_Utils::saveOrCompare("./data/testOutput/jenc_dmabuf_nv12.jpg",
        (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}


BOOST_AUTO_TEST_CASE(jpegencoder_dmabuf_fd_yuv420, * boost::unit_test::disabled())
{
    LoggerProps logProps; logProps.enableConsoleLog = true;
    Logger::initLogger(logProps);
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);

    const int width = 3840, height = 2160;
    const std::string inputYuvPath = "./data/4k.yuv";

    // Source
    auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());

    // Create DMABUF metadata with actual pitches from NvBufSurface
    auto meta = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::DMABUF));
    DMAAllocator::setMetadata(meta, width, height, ImageMetadata::YUV420);
    auto rawPin = m1->addOutputPin(meta);

    // NVJPEG encoder module
    auto enc = boost::shared_ptr<Module>(new JPEGEncoderL4TM()); // ensure this module exists/linked
    m1->setNext(enc);
    auto encMeta = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    auto encPin = enc->addOutputPin(encMeta);

    // Sink
    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    enc->setNext(sink);

    BOOST_TEST(m1->init());
    BOOST_TEST(enc->init());
    BOOST_TEST(sink->init());

    // Allocate DMABUF frame and fill with I420 data (row-wise using NvBufSurface params)
    auto frame = m1->makeFrame(meta->getDataSize(), rawPin);
    {
        auto dma = static_cast<DMAFDWrapper*>(frame->data());
        NvBufSurface* surf = dma->getNvBufSurface();

        std::ifstream in(inputYuvPath, std::ios::binary);
        BOOST_TEST(in.good());

        // planes: 0=Y,1=U,2=V
        for (int p = 0; p < 3; ++p) {
            NvBufSurfaceMap(surf, 0, p, NVBUF_MAP_READ_WRITE);
            NvBufSurfaceSyncForCpu(surf, 0, p);
            auto dst   = static_cast<uint8_t*>(surf->surfaceList[0].mappedAddr.addr[p]);
            auto rows  = surf->surfaceList[0].planeParams.height[p];
            auto pitch = surf->surfaceList[0].planeParams.pitch[p];
            auto rowB  = surf->surfaceList[0].planeParams.width[p] *
                         surf->surfaceList[0].planeParams.bytesPerPix[p];

            std::vector<uint8_t> row(rowB);
            for (uint32_t r = 0; r < rows; ++r) {
                in.read(reinterpret_cast<char*>(row.data()), rowB);
                BOOST_TEST(static_cast<size_t>(in.gcount()) == rowB);
                memcpy(dst + r * pitch, row.data(), rowB);
            }
            NvBufSurfaceSyncForDevice(surf, 0, p);
            NvBufSurfaceUnMap(surf, 0, p);
        }
    }

    frame_container fc; fc.insert({ rawPin, frame });
    m1->send(fc);
    enc->step();

    auto out = sink->pop();
    BOOST_TEST((out.find(encPin) != out.end()));
    auto outFrame = out[encPin];
    BOOST_TEST(outFrame->size() > 0);

    Test_Utils::saveOrCompare("./data/testOutput/jenc_dmabuf_yuv420.jpg",
        (const uint8_t*)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic, * boost::unit_test::disabled())
{
	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_rgb, * boost::unit_test::disabled())
{
	// metadata is known
	auto width = 1280;
	auto height = 720;
	auto fileSize = width*height*3;
	unsigned char *in_buf = new unsigned char[fileSize];

	auto in_file = new std::ifstream("./data/frame_1280x720_rgb.raw");
	in_file->read((char *)in_buf, fileSize);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());	
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));

	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/jpegencoderl4tm_frame_1280x720_rgb.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_scale, * boost::unit_test::disabled())
{
	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.scale = 0.125;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];
	BOOST_TEST(encodedImageFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm_scale_0.125.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0); 
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_rgb_perf, * boost::unit_test::disabled())
{
	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 1280;
	auto height = 720;
	FileReaderModuleProps fileReaderProps("./data/frame_1280x720_rgb.raw", 0, -1);
	fileReaderProps.fps = 1000;
	auto m1 = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));	
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps encoderProps;
	encoderProps.logHealth = true;
	encoderProps.logHealthFrequency = 100;
	auto m2 = boost::shared_ptr<Module>(new JPEGEncoderL4TM(encoderProps));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto m3 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	m2->setNext(m3);

	PipeLine p("test");
	p.appendModule(m1);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_perf, * boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.logHealth = true;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	for (auto i = 0; i < 10000; i++)
	{
		m1->send(frames);
		m2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_perf_scale, * boost::unit_test::disabled())
{

	LoggerProps logprops;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	// metadata is known
	auto width = 3840;
	auto height = 2160;
	unsigned char *in_buf = new unsigned char[width * height];

	auto in_file = new std::ifstream("./data/4k.yuv");
	in_file->read((char *)in_buf, 3840 * 2160);
	delete in_file;

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, 1, CV_8UC1, width, CV_8U));

	auto rawImagePin = m1->addOutputPin(metadata);

	JPEGEncoderL4TMProps props;
	props.logHealth = true;
	props.scale = 0.25;
	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(props));
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	//	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());
	memcpy(rawImageFrame->data(), in_buf, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	for (auto i = 0; i < 10000; i++)
	{
		m1->send(frames);
		m2->step();
		m3->pop();
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_2, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);
	m2->step();
	frames = m3->pop();
	BOOST_TEST((frames.find(encodedImagePin) != frames.end()));
	auto encodedImageFrame = frames[encodedImagePin];

	Test_Utils::saveOrCompare("./data/testOutput/frame_test_l4tm.jpg", (const uint8_t *)encodedImageFrame->data(), encodedImageFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_notmultipleof32, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img_orig = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img;
	cv::resize(img_orig, img, cv::Size(240, 60));
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);

	try
	{
		m2->step();
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_NOTIMPLEMENTED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_notmultipleof32_2, * boost::unit_test::disabled())
{
	// metadata is known
	auto img_orig = cv::imread("./data/frame.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img;
	cv::resize(img_orig, img, cv::Size(240, 60));
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	try
	{
		m1->setNext(m2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_channels_2, * boost::unit_test::disabled())
{
	// metadata is known
	auto img = cv::imread("./data/frame.jpg");
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	try
	{
		m1->setNext(m2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(jpegencoderl4tm_basic_width_channels, * boost::unit_test::disabled())
{
	// metadata is set after init
	auto img = cv::imread("./data/frame.jpg");
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	auto rawImagePin = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM());
	m1->setNext(m2);
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto encodedImagePin = m2->addOutputPin(encodedImageMetadata);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), rawImagePin);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePin, rawImageFrame));

	m1->send(frames);

	try
	{
		m2->step();
		BOOST_TEST(false);
	}
	catch (AIP_Exception &exception)
	{
		BOOST_TEST(exception.getCode() == AIP_NOTIMPLEMENTED);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_SUITE_END()
