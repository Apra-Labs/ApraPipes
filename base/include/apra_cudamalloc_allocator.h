#pragma once
#include <cstddef>

struct apra_cudamalloc_allocator
{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	static char *malloc(const size_type bytes);
	static void free(char *const block);
};