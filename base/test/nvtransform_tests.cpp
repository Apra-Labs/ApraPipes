#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "NvTransform.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "npp.h"
#include "CCKernel.h"
#include "RawImageMetadata.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "NvArgusCamera.h"
#include "test_utils.h"
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "MemTypeConversion.h"

#include <chrono>

using sys_clock = std::chrono::system_clock;

BOOST_AUTO_TEST_SUITE(nv_transform_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1920, 1080, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nv_transform);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(test, *boost::unit_test::disabled())
{
	EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(eglDisplay == EGL_NO_DISPLAY)
	{
		throw AIPException(AIP_FATAL, "eglGetDisplay failed");
	} 

	if (!eglInitialize(eglDisplay, NULL, NULL))
	{
		throw AIPException(AIP_FATAL, "eglInitialize failed");
	} 
	DMAFDWrapper* dmafdWrapper = DMAFDWrapper::create(0,1024,1024,NvBufferColorFormat_ABGR32,NvBufferLayout_Pitch,eglDisplay);
	auto mapped = dmafdWrapper->getHostPtr();
	memset(mapped,255,1024*1024*4);
	auto rgbSize = 10;
	for(auto i = 0; i < rgbSize; i++)
	{
		cout << (int)*(static_cast<uint8_t*>(mapped) + i) << " ";
	}
	cout <<endl;
}

BOOST_AUTO_TEST_CASE(fdtest, *boost::unit_test::disabled())
{
	size_t size = 1024*1024*4;
	void *host = malloc(size);
	void *host_ref = malloc(size);

	cudaFree(0);

	EGLImageKHR eglInImage;
    CUgraphicsResource  pInResource;
    CUeglFrame eglInFrame;
	EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(eglDisplay == EGL_NO_DISPLAY)
	{
		throw AIPException(AIP_FATAL, "eglGetDisplay failed");
	} 

	if (!eglInitialize(eglDisplay, NULL, NULL))
	{
		throw AIPException(AIP_FATAL, "eglInitialize failed");
	} 
	DMAFDWrapper* dmafdWrapper = DMAFDWrapper::create(0,1024,1024,NvBufferColorFormat_ABGR32,NvBufferLayout_Pitch,eglDisplay);
	auto mapped = dmafdWrapper->getHostPtr();


	auto src = DMAUtils::getCudaPtrForFD(dmafdWrapper->getFd(), eglInImage,&pInResource,eglInFrame, eglDisplay);

	int value = 128;
	for(int i = 0; i < 10; i++)
	{
		value += i;
		memset(mapped, value, size);
		NvBufferMemSyncForDevice(dmafdWrapper->getFd(), 0, &mapped);
		cudaMemcpy(host, src, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		memset(host_ref, value, size);
		if(memcmp(host, host_ref, size) != 0)
		{
			std::cout << "failed" << std::endl;
		}

		value += 1;
		cudaMemset(src, value, size);
		cudaDeviceSynchronize();
		NvBufferMemSyncForCpu(dmafdWrapper->getFd(), 0, &mapped);
		memset(host_ref, value, size);
		if(memcmp(mapped, host_ref, size) != 0)
		{
			std::cout << "failed 2" << std::endl;
		}
	}



	DMAUtils::freeCudaPtr(eglInImage,&pInResource, eglDisplay);
}

BOOST_AUTO_TEST_CASE(scaleTest, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	float scaleHeight = 2;
	float scaleWidth = 2;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 1280, 720, scaleWidth, scaleHeight, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(cropTest, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 720, 720, {0, 280, 720, 720}, NvTransformProps::NvTransformFilter::SMART, NvTransformProps::NvTransformRotate::ROTATE_0)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(scaleAndCropTest, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	float scaleHeight = 2;
	float scaleWidth = 2;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 720, 720, {0, 280, 720, 720}, scaleWidth, scaleHeight, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(scaleAndCropAndRotate180Test, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	float scaleHeight = 2;
	float scaleWidth = 2;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 720, 720, {0, 280, 720, 720}, scaleWidth, scaleHeight, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_180)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(getSetPropsTest, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	float scaleHeight = 2;
	float scaleWidth = 2;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 720, 720, {0, 280, 720, 720}, scaleWidth, scaleHeight, NvTransformProps::NvTransformFilter::BILINEAR, NvTransformProps::NvTransformRotate::ROTATE_0)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	auto propsChange = nv_transform->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_180;
	nv_transform->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(allRotateTest, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	// 1. Source 1280x720
	NvV4L2CameraProps sourceProps(1280, 720, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	// 2. Take Input from Source (1280 x 720) and apply Nv Transform to center crop source to 720 x 720
	auto nv_transform_1 = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 720, 720, {0, 280, 720, 720}, NvTransformProps::NvTransformFilter::SMART, NvTransformProps::NvTransformRotate::ROTATE_0)));
	source->setNext(nv_transform_1);

	// 3. Take Input Nv Transform 1 (720x720)and apply Nv Transform to scale to 1000 x 1000
	auto nv_transform_2 = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 1000, 1000, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	nv_transform_1->setNext(nv_transform_2);

	// 4. EGL Renderer to display real time stream
	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0,0,1)));
	nv_transform_2->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	auto propsChange = nv_transform_2->getProps();
	
	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::ROTATE_0;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_90;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_180;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_270;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::ROTATE_0;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_90;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_180;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));

	propsChange = nv_transform_2->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_270;
	nv_transform_2->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
	
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(scaleTestWithFile)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	std::vector<std::string> outFile = { "./data/testOutput/scale_8bit_frame_1280x720_rgba_to_640x360.raw" };
	Test_Utils::FileCleaner f(outFile);
	auto width = 1280;
	auto height = 720;

	auto fileReaderProps = FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(fileReaderProps)));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::MemType::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto hostToDMABUF = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(hostToDMABUF);

	auto outputWidth = 640;
	auto outputHeight = 360;
	auto nvTransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::ImageType::RGBA, outputWidth, outputHeight, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	hostToDMABUF->setNext(nvTransform);

	auto DMABUFToHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	nvTransform->setNext(DMABUFToHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], false)));
	DMABUFToHost->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();

	std::string fileComparePath = "./data/NvTransformTests/scale_8bit_frame_1280x720_rgba_to_640x360_GT.raw";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_CASE(cropTestWithFile)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	std::vector<std::string> outFile = { "./data/testOutput/crop_8bit_frame_1280x720_rgba_to_720x720.raw" };
	Test_Utils::FileCleaner f(outFile);
	auto width = 1280;
	auto height = 720;

	auto fileReaderProps = FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(fileReaderProps)));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::MemType::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto hostToDMABUF = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(hostToDMABUF);

	auto outputWidth = 720;
	auto outputHeight = 720;
	auto nvTransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::ImageType::RGBA, outputWidth, outputHeight, {0, 280, 720, 720}, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	hostToDMABUF->setNext(nvTransform);

	auto DMABUFToHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	nvTransform->setNext(DMABUFToHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], false)));
	DMABUFToHost->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();

	std::string fileComparePath = "./data/NvTransformTests/crop_8bit_frame_1280x720_rgba_to_720x720_GT.raw";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_CASE(cropAndScaleTestWithFile)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	std::vector<std::string> outFile = { "./data/testOutput/crop_8bit_frame_1280x720_rgba_to_720x720_scale_to_1280x720.raw" };
	Test_Utils::FileCleaner f(outFile);
	auto width = 1280;
	auto height = 720;

	auto fileReaderProps = FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(fileReaderProps)));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::MemType::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto hostToDMABUF = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(hostToDMABUF);

	auto outputWidth = width;
	auto outputHeight = height;
	auto nvTransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::ImageType::RGBA, outputWidth, outputHeight, {0, 280, 720, 720}, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	hostToDMABUF->setNext(nvTransform);

	auto DMABUFToHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	nvTransform->setNext(DMABUFToHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], false)));
	DMABUFToHost->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();

	std::string fileComparePath = "./data/NvTransformTests/crop_8bit_frame_1280x720_rgba_to_720x720_scale_to_1280x720_GT.raw";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_CASE(rotateTestWithFile)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	std::vector<std::string> outFile = { "./data/testOutput/rotate_8bit_frame_1280x720_rgba_to_90_degrees_clockwise.raw" };
	Test_Utils::FileCleaner f(outFile);
	auto width = 1280;
	auto height = 720;

	auto fileReaderProps = FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(fileReaderProps)));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::MemType::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto hostToDMABUF = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(hostToDMABUF);

	auto nvTransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::ImageType::RGBA, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_90)));
	hostToDMABUF->setNext(nvTransform);

	auto DMABUFToHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	nvTransform->setNext(DMABUFToHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], false)));
	DMABUFToHost->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();

	std::string fileComparePath = "./data/NvTransformTests/rotate_8bit_frame_1280x720_rgba_to_90_degrees_clockwise_GT.raw";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_CASE(getSetPropsTestWithFile)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	std::vector<std::string> outFile = { "./data/testOutput/rotate_8bit_frame_1280x720_rgba_to_90_degrees_clockwise_using_get_set_props.raw" };
	Test_Utils::FileCleaner f(outFile);
	auto width = 1280;
	auto height = 720;

	auto fileReaderProps = FileReaderModuleProps("./data/8bit_frame_1280x720_rgba.raw");
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(fileReaderProps)));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::MemType::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto hostToDMABUF = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::DMABUF)));
	fileReader->setNext(hostToDMABUF);

	auto nvTransform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::ImageType::RGBA, NvTransformProps::NvTransformFilter::NICEST, NvTransformProps::NvTransformRotate::ROTATE_0)));
	hostToDMABUF->setNext(nvTransform);

	auto DMABUFToHost = boost::shared_ptr<Module>(new MemTypeConversion(MemTypeConversionProps(FrameMetadata::HOST)));
	nvTransform->setNext(DMABUFToHost);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(outFile[0], false)));
	DMABUFToHost->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	auto propsChange = nvTransform->getProps();
	propsChange.rotateType = NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_90;
	nvTransform->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	p.stop();
	p.term();
	p.wait_for_all();

	std::string fileComparePath = "./data/NvTransformTests/rotate_8bit_frame_1280x720_rgba_to_90_degrees_clockwise_using_get_set_props_GT.raw";
	uint8_t* frameData;
	uint frameSize;
	Test_Utils::readFile(outFile[0], (const uint8_t*&)frameData, frameSize);

	Test_Utils::saveOrCompare(fileComparePath.c_str(), (const unsigned char*)frameData, (size_t)frameSize, 0);
}

BOOST_AUTO_TEST_SUITE_END()