#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <boost/test/unit_test.hpp>

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(unit_tests)

BOOST_AUTO_TEST_CASE(egltest)
{
	// egldisplay -
	EGLDisplay eglDisplay;
	eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(eglDisplay == EGL_NO_DISPLAY)
	{
		throw AIPException(AIP_FATAL, "eglGetDisplay failed");
	} 

	if (!eglInitialize(eglDisplay, NULL, NULL))
	{
		throw AIPException(AIP_FATAL, "eglInitialize failed");
	}
	// cudafree
	cudaFree(0);
	// create fd
	 NvBufferCreateParams inputParams = {0};

    inputParams.width = 640;
    inputParams.height = 480;
    inputParams.layout =  NvBufferLayout_Pitch;
    inputParams.colorFormat = NvBufferColorFormat_UYVY;
    inputParams.payloadType = NvBufferPayload_SurfArray;
    inputParams.nvbuf_tag = NvBufferTag_CAMERA;

	int fd = -1;
    if (NvBufferCreateEx(&fd, &inputParams))
	{
		LOG_ERROR << "Failed to create Buffer";
	}
	auto eglImage = NvEGLImageFromFd(eglDisplay, fd);
    if (eglImage == EGL_NO_IMAGE_KHR)
    {
        LOG_ERROR << "Failed to create EGLImage";        
    }
	CUgraphicsResource pResource;

	auto status = cuGraphicsEGLRegisterImage(&pResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
	if (status != CUDA_SUCCESS)
	{
		LOG_ERROR << "cuGraphicsEGLRegisterImage failed: " << status << " cuda process stop";
	}

}

BOOST_AUTO_TEST_SUITE_END()