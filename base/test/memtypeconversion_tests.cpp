#include <boost/test/unit_test.hpp>
#include "MemTypeConversion.h"
#include "FileReaderModule.h"
#include "Logger.h"
#include "PipeLine.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FileWriterModule.h"

#if defined(__arm__) || defined(__aarch64__)
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "EglRenderer.h"
#endif

BOOST_AUTO_TEST_SUITE(memtypeconversion_tests)

BOOST_AUTO_TEST_CASE(Host_to_Dma)
{
	#if defined(__arm__) || defined(__aarch64__)
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(memconversion);

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Host_to_Dma_Planar)
{
	#if defined(__arm__) || defined(__aarch64__)
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    fileReader->addOutputPin(metadata);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF,stream)));
	fileReader->setNext(memconversion);

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Dma_to_Host,*boost::unit_test::disabled())
{
	#if defined(__arm__) || defined(__aarch64__)
    NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::YUV420));
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

BOOST_AUTO_TEST_CASE(Dma_to_Host_to_Dma,*boost::unit_test::disabled())
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

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Device_to_Dma_BGRA)
{   
	#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_400x400_BGRA.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(400,400, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	fileReader->setNext(copy1);

    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF,stream)));
	copy1->setNext(memconversion);

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Device_to_Dma_RGBA)
{   
	#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw")));
    auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
    fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	fileReader->setNext(copy1);

    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF,stream)));
	copy1->setNext(memconversion);

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Device_to_Dma_Planar)
{   
	#if defined(__arm__) || defined(__aarch64__)
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_400x400.raw")));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(400,400, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	fileReader->setNext(copy1);

    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF,stream)));
	copy1->setNext(memconversion);

    auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0,0)));
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

BOOST_AUTO_TEST_CASE(Dma_to_Device_Planar,*boost::unit_test::disabled())
{   
	#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));
    
	auto transform = boost::shared_ptr<Module>(new NvTransform(ImageMetadata::NV12));
	source->setNext(transform);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	transform->setNext(memconversion);

	auto copy2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST,stream)));
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

BOOST_AUTO_TEST_CASE(Dma_to_Device,*boost::unit_test::disabled())
{   
	#if defined(__arm__) || defined(__aarch64__)
	NvV4L2CameraProps nvCamProps(640, 360, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));
    
    auto stream = cudastream_sp(new ApraCudaStream);
    auto memconversion = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	source->setNext(memconversion);

	auto copy2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST,stream)));
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
    auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	fileReader->setNext(memconversion1);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST,stream)));
	memconversion1->setNext(memconversion2);

    auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frame_????.raw")));
	memconversion2->setNext(sink);

    PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all(); 
}

BOOST_AUTO_TEST_CASE(Host_to_Device_to_Host_PlanarImage)
{   
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/nv12-704x576.raw")));
    auto metadata = framemetadata_sp(new RawImagePlanarMetadata(704, 576, ImageMetadata::ImageType::NV12, size_t(0), CV_8U));
    fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
    auto memconversion1 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::CUDA_DEVICE,stream)));
	fileReader->setNext(memconversion1);

	auto memconversion2 = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST,stream)));
	memconversion1->setNext(memconversion2);

    auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/frame_????.raw")));
	memconversion2->setNext(sink);

    PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()