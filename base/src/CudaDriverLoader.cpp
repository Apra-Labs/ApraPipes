#include "CudaDriverLoader.h"
#include "Logger.h"
#include <sstream>

#ifdef _WIN32
// Helper to get Windows error message
static std::string getWindowsError() {
    DWORD error = GetLastError();
    if (error == 0) {
        return "Unknown error";
    }
    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&messageBuffer, 0, NULL);
    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    // Trim trailing newlines
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    return message;
}
#endif

CudaDriverLoader::CudaDriverLoader() {
    // Try to load CUDA driver library at runtime
    // On systems without NVIDIA driver (e.g., GitHub runners), this will fail gracefully
#ifdef _WIN32
    // Windows: Load nvcuda.dll
    libHandle = LoadLibraryA("nvcuda.dll");

    if (!libHandle) {
        errorMessage = "nvcuda.dll not found: " + getWindowsError();
        LOG_INFO << "CUDA driver library not available: " << errorMessage;
        LOG_INFO << "GPU-accelerated modules will be disabled. CPU fallbacks will be used.";
        return;
    }
#else
    // Linux/Jetson: Load libcuda.so.1
    libHandle = dlopen("libcuda.so.1", RTLD_LAZY);

    if (!libHandle) {
        // Library not found - this is expected on systems without GPU/driver
        const char* error = dlerror();
        errorMessage = error ? error : "libcuda.so.1 not found";
        LOG_INFO << "CUDA driver library not available: " << errorMessage;
        LOG_INFO << "GPU-accelerated modules will be disabled. CPU fallbacks will be used.";
        return;
    }
#endif

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

#ifndef _WIN32
    // EGL-based graphics interop - Linux/Jetson only
    loadSymbol(cuGraphicsEGLRegisterImage, "cuGraphicsEGLRegisterImage");
    loadSymbol(cuGraphicsResourceGetMappedEglFrame, "cuGraphicsResourceGetMappedEglFrame");
    loadSymbol(cuGraphicsUnregisterResource, "cuGraphicsUnregisterResource");
#endif

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
#ifdef _WIN32
        FreeLibrary(libHandle);
#else
        dlclose(libHandle);
#endif
        libHandle = nullptr;
        return;
    }

    LOG_INFO << "CUDA driver library loaded successfully. GPU acceleration available.";
}

CudaDriverLoader::~CudaDriverLoader() {
    if (libHandle) {
#ifdef _WIN32
        FreeLibrary(libHandle);
#else
        dlclose(libHandle);
#endif
        libHandle = nullptr;
    }
}
