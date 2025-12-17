#pragma once

#include "cuda.h"
#include "Logger.h"
#include "cuda_runtime_api.h"
#include "CudaDriverLoader.h"

inline bool check(CUresult e, int iLine, const char *szFile)
{
    if (e != CUDA_SUCCESS)
    {
        const char *szErrName = NULL;
        auto& loader = CudaDriverLoader::getInstance();
        if (loader.cuGetErrorName) {
            loader.cuGetErrorName(e, &szErrName);
        }
        LOG_FATAL << "CUDA driver API error " << (szErrName ? szErrName : "Unknown") << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

class CudaUtils
{
public:
    static bool isCudaSupported();
};

#define ck(call) check(call, __LINE__, __FILE__)

class ApraCUcontext
{
public:
    ApraCUcontext()
    {
        auto& loader = CudaDriverLoader::getInstance();
        if (!loader.isAvailable()) {
            throw AIPException(AIP_NOTEXEPCTED, "ApraCUcontext requires CUDA driver but libcuda.so not available. Error: " + loader.getErrorMessage());
        }

        ck(loader.cuInit(0));
        int nGpu = 0;
        ck(loader.cuDeviceGetCount(&nGpu));
        m_cuDevice = 0;
        ck(loader.cuDeviceGet(&m_cuDevice, 0));
        char szDeviceName[80];
        ck(loader.cuDeviceGetName(szDeviceName, sizeof(szDeviceName), m_cuDevice));
        LOG_INFO << "GPU "<<nGpu<<" in use: " << szDeviceName;

        ck(loader.cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice));
    }

    ~ApraCUcontext()
    {
        auto& loader = CudaDriverLoader::getInstance();
        if (loader.isAvailable() && loader.cuDevicePrimaryCtxRelease) {
            ck(loader.cuDevicePrimaryCtxRelease(m_cuDevice));
        }
    }

    CUcontext getContext()
    {
        return m_cuContext;
    }
    bool getComputeCapability(int& major, int& minor)
    {
        auto rc1=cudaDeviceGetAttribute(& major, ::cudaDevAttrComputeCapabilityMajor, 0);
        auto rc2=cudaDeviceGetAttribute(& minor, ::cudaDevAttrComputeCapabilityMinor, 0);
        return (rc1==::cudaSuccess && rc2==::cudaSuccess);
    }

private:
    CUcontext m_cuContext;
    CUdevice m_cuDevice;
};

class ApraCudaStream
{
public:
    ApraCudaStream()
    {
        cudaStreamCreate(&m_stream);
    }

    ~ApraCudaStream()
    {
        cudaStreamDestroy(m_stream);
    }

    cudaStream_t getCudaStream()
    {
        return m_stream;
    }

private:
    cudaStream_t m_stream;
};

typedef boost::shared_ptr<ApraCUcontext> apracucontext_sp;
typedef boost::shared_ptr<ApraCudaStream> cudastream_sp;