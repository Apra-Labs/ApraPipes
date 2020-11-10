#include "apra_cudamallochost_allocator.h"


	char* apra_cudamallochost_allocator::malloc(const size_type bytes)
	{
		void *ptr;
		auto errorCode = cudaMallocHost(&ptr, bytes);

		if (errorCode != cudaSuccess)
		{
			// failed to allocate memory
			return NULL;
		}

		return reinterpret_cast<char *>(ptr);
	}
	
	void apra_cudamallochost_allocator::free(char *const block)
	{
		auto errorCode = cudaFreeHost(block);
		if (errorCode != cudaSuccess)
		{
			// log error
		}
	}