#pragma once

#include <cstdlib>

#ifdef APRA_CUDA_ENABLED
    #include <boost/pool/pool.hpp>
    #include "ApraPool.h"
    #include "apra_cudamallochost_allocator.h"
    #include "apra_cudamalloc_allocator.h"
#endif

// HostAllocator
//
// Previously this class was backed by boost::pool<>::ordered_malloc/ordered_free.
// That had two bad properties for a long-running streaming workload:
//   1. boost::pool only returns memory to the OS in its destructor (via
//      release_memory()) — during the run the pool's free-list grows but its
//      RSS only ever rises. Combined with next_size doubling, a single pool
//      block could grow to hundreds of megabytes that the OS never reclaims.
//   2. ordered_malloc/ordered_free are O(N) over the free-list, so per-frame
//      CPU also climbed with the pool size.
//
// FrameFactory now matches each allocateChunks() 1:1 with a single freeChunks()
// of the same head pointer (no more mid-buffer partial frees), so a plain
// std::malloc / std::free is sufficient and lets glibc reclaim memory the way
// it normally would.
class HostAllocator
{
public:
    HostAllocator() = default;
    virtual ~HostAllocator() = default;

    virtual void *allocateChunks(size_t n)
    {
        return std::malloc(n * getChunkSize());
    }

    // n is unused: with std::malloc/std::free the size is tracked by the
    // allocator. The parameter is kept for ABI compatibility with the
    // pre-existing virtual signature shared with the CUDA variants and
    // DMAAllocator.
    virtual void freeChunks(void *MemPtr, size_t /*n*/)
    {
        std::free(MemPtr);
    }

    virtual size_t getChunkSize()
    {
        return 1024;
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