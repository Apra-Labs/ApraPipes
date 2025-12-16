#include "CudaDriverLoader.h"
#include "Logger.h"
#include <sstream>

CudaDriverLoader::CudaDriverLoader() {
    // Try to load libcuda.so.1 at runtime
    // On systems without NVIDIA driver (e.g., GitHub runners), this will fail gracefully
    libHandle = dlopen("libcuda.so.1", RTLD_LAZY);

    if (!libHandle) {
        // Library not found - this is expected on systems without GPU/driver
        const char* error = dlerror();
        errorMessage = error ? error : "libcuda.so.1 not found";
        LOG_INFO << "CUDA driver library not available: " << errorMessage;
        LOG_INFO << "GPU-accelerated modules will be disabled. CPU fallbacks will be used.";
        return;
    }

    LOG_INFO << "Loading CUDA driver library symbols...";

    // Load all function symbols
    loadSymbol(cuInit, "cuInit");
    loadSymbol(cuDeviceGet, "cuDeviceGet");
    loadSymbol(cuDeviceGetCount, "cuDeviceGetCount");
    loadSymbol(cuDeviceGetName, "cuDeviceGetName");

    loadSymbol(cuCtxCreate, "cuCtxCreate");
    loadSymbol(cuCtxDestroy, "cuCtxDestroy");
    loadSymbol(cuCtxPushCurrent, "cuCtxPushCurrent");
    loadSymbol(cuCtxPopCurrent, "cuCtxPopCurrent");
    loadSymbol(cuCtxSynchronize, "cuCtxSynchronize");
    loadSymbol(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    loadSymbol(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease");

    loadSymbol(cuMemAlloc, "cuMemAlloc");
    loadSymbol(cuMemAllocPitch, "cuMemAllocPitch");
    loadSymbol(cuMemFree, "cuMemFree");
    loadSymbol(cuMemcpy2D, "cuMemcpy2D");
    loadSymbol(cuMemcpy2DAsync, "cuMemcpy2DAsync");

    loadSymbol(cuStreamCreate, "cuStreamCreate");
    loadSymbol(cuStreamDestroy, "cuStreamDestroy");
    loadSymbol(cuStreamSynchronize, "cuStreamSynchronize");

    loadSymbol(cuGraphicsEGLRegisterImage, "cuGraphicsEGLRegisterImage");
    loadSymbol(cuGraphicsResourceGetMappedEglFrame, "cuGraphicsResourceGetMappedEglFrame");
    loadSymbol(cuGraphicsUnregisterResource, "cuGraphicsUnregisterResource");

    loadSymbol(cuGetErrorName, "cuGetErrorName");
    loadSymbol(cuGetErrorString, "cuGetErrorString");

    // Verify critical symbols loaded successfully
    if (!cuInit || !cuDeviceGetCount || !cuCtxCreate) {
        std::ostringstream oss;
        oss << "Failed to load critical CUDA driver symbols: ";
        if (!cuInit) oss << "cuInit ";
        if (!cuDeviceGetCount) oss << "cuDeviceGetCount ";
        if (!cuCtxCreate) oss << "cuCtxCreate ";

        errorMessage = oss.str();
        LOG_ERROR << errorMessage;

        // Close library handle since we can't use it
        dlclose(libHandle);
        libHandle = nullptr;
        return;
    }

    LOG_INFO << "CUDA driver library loaded successfully. GPU acceleration available.";
}

CudaDriverLoader::~CudaDriverLoader() {
    if (libHandle) {
        dlclose(libHandle);
        libHandle = nullptr;
    }
}
