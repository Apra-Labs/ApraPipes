#pragma once

#include "cuda.h"
#include "Logger.h"
#include "cuda_runtime_api.h"

inline bool check(CUresult e, int iLine, const char *szFile)
{
    if (e != CUDA_SUCCESS)
    {
        const char *szErrName = NULL;
        cuGetErrorName(e, &szErrName);
        LOG_FATAL << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile;
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
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        m_cuDevice = 0;
        ck(cuDeviceGet(&m_cuDevice, 0));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), m_cuDevice));
        LOG_INFO << "GPU in use: " << szDeviceName;

        ck(cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice));
    }

    ~ApraCUcontext()
    {
        ck(cuDevicePrimaryCtxRelease(m_cuDevice));
    }

    CUcontext getContext()
    {
        return m_cuContext;
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