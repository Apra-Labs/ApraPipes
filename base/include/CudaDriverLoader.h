#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // Prevent Windows.h from defining min/max macros that conflict with std::min/max
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda.h>
#ifndef _WIN32
// EGL is only available on Linux/Jetson
#include <cudaEGL.h>
#endif
#include <string>

/**
 * @brief Singleton class that dynamically loads libcuda.so (Linux) or nvcuda.dll (Windows) at runtime
 *
 * This allows the executable to start even when the CUDA driver is not available.
 * When GPU is not present (e.g., GitHub CI runners), the library fails to load gracefully,
 * and isAvailable() returns false. Modules requiring CUDA can check availability and throw
 * appropriate exceptions.
 *
 * Platform-specific behavior:
 * - Linux: Uses dlopen/dlsym to load libcuda.so.1
 * - Windows: Uses LoadLibrary/GetProcAddress to load nvcuda.dll
 *
 * Usage:
 *   auto& loader = CudaDriverLoader::getInstance();
 *   if (!loader.isAvailable()) {
 *       throw AIPException(AIP_NOTEXEPCTED, "CUDA driver not available");
 *   }
 *   CUresult result = loader.cuInit(0);
 */
class CudaDriverLoader {
public:
    static CudaDriverLoader& getInstance() {
        static CudaDriverLoader instance;
        return instance;
    }

    // Prevent copying
    CudaDriverLoader(const CudaDriverLoader&) = delete;
    CudaDriverLoader& operator=(const CudaDriverLoader&) = delete;

    bool isAvailable() const { return libHandle != nullptr; }

    const std::string& getErrorMessage() const { return errorMessage; }

    // CUDA Driver API function pointers
    // Initialization
    CUresult (*cuInit)(unsigned int Flags) = nullptr;

    // Device management
    CUresult (*cuDeviceGet)(CUdevice *device, int ordinal) = nullptr;
    CUresult (*cuDeviceGetCount)(int *count) = nullptr;
    CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev) = nullptr;

    // Context management
    CUresult (*cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext ctx) = nullptr;
    CUresult (*cuCtxPushCurrent)(CUcontext ctx) = nullptr;
    CUresult (*cuCtxPopCurrent)(CUcontext *pctx) = nullptr;
    CUresult (*cuCtxSynchronize)(void) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *pctx, CUdevice dev) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice dev) = nullptr;

    // Memory management
    CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize) = nullptr;
    CUresult (*cuMemAllocPitch)(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr dptr) = nullptr;
    CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D *pCopy) = nullptr;
    CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *pCopy, CUstream hStream) = nullptr;

    // Stream management
    CUresult (*cuStreamCreate)(CUstream *phStream, unsigned int Flags) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream hStream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream hStream) = nullptr;

#ifndef _WIN32
    // Graphics interop (for DMAUtils) - Linux/Jetson only (EGL-based)
    CUresult (*cuGraphicsEGLRegisterImage)(CUgraphicsResource *pCudaResource, EGLImageKHR image, unsigned int flags) = nullptr;
    CUresult (*cuGraphicsResourceGetMappedEglFrame)(CUeglFrame *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) = nullptr;
    CUresult (*cuGraphicsUnregisterResource)(CUgraphicsResource resource) = nullptr;
#endif

    // Error handling
    CUresult (*cuGetErrorName)(CUresult error, const char **pStr) = nullptr;
    CUresult (*cuGetErrorString)(CUresult error, const char **pStr) = nullptr;

private:
    CudaDriverLoader();
    ~CudaDriverLoader();

#ifdef _WIN32
    HMODULE libHandle = nullptr;
#else
    void* libHandle = nullptr;
#endif
    std::string errorMessage;

    template<typename FuncPtr>
    void loadSymbol(FuncPtr& funcPtr, const char* symbolName) {
#ifdef _WIN32
        funcPtr = reinterpret_cast<FuncPtr>(GetProcAddress(libHandle, symbolName));
#else
        funcPtr = reinterpret_cast<FuncPtr>(dlsym(libHandle, symbolName));
#endif
        if (!funcPtr) {
            // Not all symbols may be available in all CUDA versions
            // We'll log but not fail initialization
        }
    }
};
