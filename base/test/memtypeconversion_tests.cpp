#include <boost/test/unit_test.hpp>
#include "MemTypeConversion.h"
#include "FileReaderModule.h"
#include "Logger.h"
#include "PipeLine.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FileWriterModule.h"
#include "ResizeNPPI.h"
#include "test_utils.h"
#include "ExternalSinkModule.h"

#if defined(__arm__) || defined(__aarch64__)
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "EglRenderer.h"
#endif

BOOST_AUTO_TEST_SUITE(memtypeconversion_tests)

BOOST_AUTO_TEST_CASE(Host_to_Dma_to_Device_to_Host_RGBA_1280x720)
{
#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(memconversion1);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	memconversion1->setNext(memconversion2);

	auto memconversion3 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversion2->setNext(memconversion3);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	memconversion3->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(memconversion1->init());
	BOOST_TEST(memconversion2->init());
	BOOST_TEST(memconversion3->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	memconversion1->step();
	memconversion2->step();
	memconversion3->step();

	auto outputPinId = memconversion3->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/MemConversion_outputs/Host_to_Dma_to_Device_to_Host_RGBA_1280x720.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
#endif
}

BOOST_AUTO_TEST_CASE(Host_to_Device_to_Dma_to_Device_to_Host_YUV420_400x400)
{
#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_400x400.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(400, 400, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(memconversion1);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	memconversion1->setNext(memconversion2);

	auto memconversion3 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	memconversion2->setNext(memconversion3);

	auto memconversion4 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversion3->setNext(memconversion4);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	memconversion4->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(memconversion1->init());
	BOOST_TEST(memconversion2->init());
	BOOST_TEST(memconversion3->init());
	BOOST_TEST(memconversion4->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	memconversion1->step();
	memconversion2->step();
	memconversion3->step();
	memconversion4->step();

	auto outputPinId = memconversion4->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/MemConversion_outputs/Host_to_Device_to_Dma_to_Device_to_Host_YUV420_400x400.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
#endif
}

BOOST_AUTO_TEST_CASE(Host_to_Device_to_Dma_to_Host_BGRA_400x400)
{
#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_400x400_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(400, 400, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(memconversion1);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	memconversion1->setNext(memconversion2);

	auto memconversion3 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	memconversion2->setNext(memconversion3);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	memconversion3->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(memconversion1->init());
	BOOST_TEST(memconversion2->init());
	BOOST_TEST(memconversion3->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	memconversion1->step();
	memconversion2->step();
	memconversion3->step();

	auto outputPinId = memconversion3->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/MemConversion_outputs/Host_to_Device_to_Dma_to_Host_BGRA_400x400.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
#endif
}

BOOST_AUTO_TEST_CASE(Dma_to_Host, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::RGBA));
	source->setNext(transform);

	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	transform->setNext(memconversion);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/frame_????.raw")));
	memconversion->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_CASE(Dma_to_Host_to_Dma, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::RGBA));
	source->setNext(transform);

	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	transform->setNext(memconversion);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	memconversion->setNext(memconversion2);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0, 0, 0)));
	memconversion2->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(120));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
#endif
}

BOOST_AUTO_TEST_CASE(Device_to_Dma_RGBA, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(copy1);

	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	copy1->setNext(memconversion);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0, 0, 0)));
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

BOOST_AUTO_TEST_CASE(Device_to_Dma_Planar, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_400x400.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(400, 400, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(copy1);

	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF, stream)));
	copy1->setNext(memconversion);

	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0, 0, 0)));
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

BOOST_AUTO_TEST_CASE(Dma_to_Device_Planar, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::NV12));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	transform->setNext(memconversion);

	auto copy2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversion->setNext(copy2);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/frame_????.raw")));
	copy2->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
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

BOOST_AUTO_TEST_CASE(Dma_to_Device, *boost::unit_test::disabled())
{
#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	source->setNext(memconversion);

	auto copy2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversion->setNext(copy2);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/frame_????.raw")));
	copy2->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
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

BOOST_AUTO_TEST_CASE(Host_to_Device_to_Host)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/RGB_320x180.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(320, 180, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(memconversion1);

	auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(400, 400, stream)));
	memconversion1->setNext(resize);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	resize->setNext(memconversion2);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	memconversion2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(resize->init());
	BOOST_TEST(memconversion1->init());
	BOOST_TEST(memconversion2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	memconversion1->step();
	resize->step();
	memconversion2->step();

	auto outputPinId = memconversion2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/MemConversion_outputs/Host_to_Device_to_Host_RGB.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(Host_to_Device_to_Host_PlanarImage)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/nv12-704x576.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(704, 576, ImageMetadata::ImageType::NV12, size_t(0), CV_8U));
	fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE, stream)));
	fileReader->setNext(memconversion1);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST, stream)));
	memconversion1->setNext(memconversion2);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	memconversion2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(memconversion1->init());
	BOOST_TEST(memconversion2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	memconversion1->step();
	memconversion2->step();

	auto outputPinId = memconversion2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE_PLANAR)[0];
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/MemConversion_outputs/Host_to_Device_to_Host_NV12.raw", (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()