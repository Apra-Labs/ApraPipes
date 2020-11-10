#include "apra_cudamalloc_allocator.h"



	char* apra_cudamalloc_allocator::malloc(const size_type bytes)
	{
		void *ptr;
		auto errorCode = cudaMalloc(&ptr, bytes);

		if (errorCode != cudaSuccess)
		{
			// failed to allocate memory
			return NULL;
		}

		return reinterpret_cast<char *>(ptr);
	}
	
	void apra_cudamalloc_allocator::free(char *const block)
	{
		auto errorCode = cudaFree(block);
		if (errorCode != cudaSuccess)
		{
			// log error
		}
	}