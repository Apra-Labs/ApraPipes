#pragma once

#include <boost/pool/pool.hpp>

#ifdef APRA_CUDA_ENABLED
    #include "ApraPool.h"
    #include "apra_cudamallochost_allocator.h"
    #include "apra_cudamalloc_allocator.h"
#endif

class HostAllocator
{
protected:
    boost::pool<> buff_allocator;

public:
    HostAllocator() : buff_allocator(512) {};
    virtual ~HostAllocator()
    {
        buff_allocator.release_memory();
    }
    virtual void *allocateChunks(size_t n)
    {   void *ptr = buff_allocator.ordered_malloc(n);
        // LOG_ERROR << "Allocated " << n << " chunks at " << ptr;
        return ptr;
    }

    virtual void freeChunks(void *MemPtr, size_t n)
    {
        // LOG_ERROR << "Freeing " << n << " chunks at " << MemPtr;
        buff_allocator.ordered_free(MemPtr, n);
    }

    virtual size_t getChunkSize()
    {
        return 512;
    }
};

#ifdef APRA_CUDA_ENABLED
class HostPinnedAllocator : public HostAllocator
{
protected:
    boost::pool<apra_cudamallochost_allocator> buff_pinned_allocator;

public:
    HostPinnedAllocator() : buff_pinned_allocator(1024) {};
    ~HostPinnedAllocator()
    {
        buff_pinned_allocator.release_memory();
    }
    void *allocateChunks(size_t n)
    {
        return buff_pinned_allocator.ordered_malloc(n);
    }
    void freeChunks(void *MemPtr, size_t n)
    {
        buff_pinned_allocator.ordered_free(MemPtr, n);
    }
};

class CudaDeviceAllocator : public HostAllocator
{
protected:
    ApraPool<apra_cudamalloc_allocator> buff_cudadevice_allocator;

public:
    CudaDeviceAllocator() : buff_cudadevice_allocator(1024) {};
    ~CudaDeviceAllocator()
    {
        buff_cudadevice_allocator.release_memory();
    }
    void *allocateChunks(size_t n)
    {
        return buff_cudadevice_allocator.ordered_malloc(n);
    }
    void freeChunks(void *MemPtr, size_t n)
    {
        buff_cudadevice_allocator.ordered_free(MemPtr, n);
    }
};
#endif