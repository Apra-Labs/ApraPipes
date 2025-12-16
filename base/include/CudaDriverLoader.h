#pragma once

#include <dlfcn.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <string>

/**
 * @brief Singleton class that dynamically loads libcuda.so at runtime using dlopen/dlsym
 *
 * This allows the executable to start even when libcuda.so (NVIDIA driver) is not available.
 * When GPU is not present (e.g., GitHub CI runners), the library fails to load gracefully,
 * and isAvailable() returns false. Modules requiring CUDA can check availability and throw
 * appropriate exceptions.
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

    // Graphics interop (for DMAUtils)
    CUresult (*cuGraphicsEGLRegisterImage)(CUgraphicsResource *pCudaResource, EGLImageKHR image, unsigned int flags) = nullptr;
    CUresult (*cuGraphicsResourceGetMappedEglFrame)(CUeglFrame *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) = nullptr;
    CUresult (*cuGraphicsUnregisterResource)(CUgraphicsResource resource) = nullptr;

    // Error handling
    CUresult (*cuGetErrorName)(CUresult error, const char **pStr) = nullptr;
    CUresult (*cuGetErrorString)(CUresult error, const char **pStr) = nullptr;

private:
    CudaDriverLoader();
    ~CudaDriverLoader();

    void* libHandle = nullptr;
    std::string errorMessage;

    template<typename FuncPtr>
    void loadSymbol(FuncPtr& funcPtr, const char* symbolName) {
        funcPtr = reinterpret_cast<FuncPtr>(dlsym(libHandle, symbolName));
        if (!funcPtr) {
            // Not all symbols may be available in all CUDA versions
            // We'll log but not fail initialization
        }
    }
};
