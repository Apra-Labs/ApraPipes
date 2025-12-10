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

#include <chrono>

using sys_clock = std::chrono::system_clock;

BOOST_AUTO_TEST_SUITE(nv_transform_tests, *boost::unit_test::disabled())

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

BOOST_AUTO_TEST_CASE(test)
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

BOOST_AUTO_TEST_CASE(fdtest)
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

BOOST_AUTO_TEST_CASE(crop, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvArgusCameraProps sourceProps(1280, 720, 0);
	auto source = boost::shared_ptr<Module>(new NvArgusCamera(sourceProps));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 400, 400)));
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

BOOST_AUTO_TEST_CASE(cropAndScale, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1920, 1080, 10);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	float scaleHeight = 1;
	float scaleWidth = 0.5;
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA, 400, 400, scaleWidth, scaleHeight, NvTransformProps::NvTransformFilter::BILINEAR)));
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

BOOST_AUTO_TEST_SUITE_END()