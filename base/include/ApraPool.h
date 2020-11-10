#pragma once

#include<stddef.h>
#include <functional>   // std::greater
#include <boost/pool/object_pool.hpp>

class ApraChunk
{
public:
	ApraChunk() :ptr(nullptr), next(nullptr)
	{

	}

	void set(void* _ptr, ApraChunk* _next)
	{
		ptr = _ptr;
		next = _next;
	}

	void setNext(ApraChunk* _next)
	{
		next = _next;
	}

	void* get()
	{
		return ptr;
	}

	ApraChunk* getNext()
	{
		return next;
	}

private:
	void* ptr;
	ApraChunk* next;
};

class ApraSegregatedStorage
{
public:
	ApraSegregatedStorage();
	virtual ~ApraSegregatedStorage();	

	void* malloc_n(const size_t n, const size_t partition_size);
	void add_ordered_block(void* const block, const size_t nsz, const size_t npartition_sz);
	void ordered_free_n(void* const chunks, const size_t n, const size_t partition_size);

protected:
	bool empty() const;	

//private:
	ApraChunk* segregate(void* block, size_t sz, size_t partition_sz, ApraChunk* end = nullptr);
	ApraChunk* find_prev(void * const ptr);
	ApraChunk* try_malloc_n(ApraChunk*& start, size_t n, const size_t partition_size);

	ApraChunk* getFreeApraChunk();
	void addToJunk(ApraChunk* start, ApraChunk* end);	
	void releaseChunks();

	ApraChunk* first;
	ApraChunk* junkList; // used to store unused ApraChunks
	boost::object_pool<ApraChunk> chunk_opool; // object pool takes c are of deleting the memory
};

class ApraNode
{
private:
	char * ptr;
	char* end_ptr;
	size_t sz;

	ApraNode* next;

public:
	ApraNode(char * const nptr, const size_t nsize)
		:ptr(nptr), sz(nsize), end_ptr(nptr + nsize), next(nullptr)
	{
	}
	
	~ApraNode()
	{
		ptr = nullptr;
		next = nullptr;
		end_ptr = nullptr;
	}

	bool valid() const
	{
		return (begin() != 0);
	}
	void invalidate()
	{
		begin() = 0;
	}
	char * & begin()
	{
		return ptr;
	}
	char * begin() const
	{
		return ptr;
	}
	char * end() const
	{
		return end_ptr;
	}
	size_t total_size() const
	{
		return sz;
	}

	ApraNode* getNext()
	{
		return next;
	}

	void setNext(ApraNode* _next)
	{
		next = _next;
	}
};

template <typename UserAllocator>
class ApraPool : protected ApraSegregatedStorage
{
public:	
	explicit ApraPool(const size_t nRequestedSize, const size_t nnext_size = 32, const size_t nmax_size = 0);
	virtual ~ApraPool();

	void* ordered_malloc(const size_t n);
	void ordered_free(void* ptr, const size_t n);

	bool release_memory();
	bool purge_memory();
private:
	
	bool is_from(void* const chunk, char* const start, char* const end);

	ApraSegregatedStorage & store()
	{
		return *this;
	}
	const ApraSegregatedStorage & store() const
	{
		return *this;
	}

	const size_t requested_size;
	size_t start_size;
	size_t next_size;
	size_t max_size;

	ApraNode* list;
	boost::object_pool<ApraNode> node_opool;
};

template <typename UserAllocator>
ApraPool<UserAllocator>::ApraPool(const size_t nRequestedSize, const size_t nnext_size, const size_t nmax_size) : requested_size(nRequestedSize), start_size(nnext_size), next_size(nnext_size), max_size(nmax_size), list(nullptr)
{

}

template <typename UserAllocator>
ApraPool<UserAllocator>::~ApraPool()
{ //!   Destructs the Pool, freeing its list of memory blocks.
	purge_memory();
}

template <typename UserAllocator>
bool ApraPool<UserAllocator>::purge_memory()
{
	if (list == nullptr)
	{
		return false;
	}

	ApraNode* prev = nullptr;
	while (list != nullptr)
	{
		// delete the storage
		(UserAllocator::free)(list->begin());
		prev = list;
		list = list->getNext();
		node_opool.destroy(prev);
	}

	this->releaseChunks();
	next_size = start_size;

	return true;
}

template <typename UserAllocator>
bool ApraPool<UserAllocator>::release_memory()
{
	bool ret = false;

	ApraNode* ptr = list;
	ApraNode* prev = nullptr;

	ApraChunk* free_chunk = this->first;
	ApraChunk* prev_free_chunk = nullptr;

	while (ptr != nullptr)
	{
		ApraNode& node = *ptr;
		if (free_chunk == nullptr)
		{
			break;
		}

		bool all_chunks_free = true;

		ApraChunk* saved_free = free_chunk;
		ApraChunk* last_free = nullptr;
		for (char * i = node.begin(); i != node.end(); i += requested_size)
		{
			if (i != free_chunk->get())
			{
				all_chunks_free = false;

				free_chunk = saved_free;
				break;
			}

			if (i + requested_size == node.end())
			{
				// all chunks are free
				last_free = free_chunk;
			}

			free_chunk = free_chunk->getNext();
		}

		ApraNode* next = ptr->getNext();

		if (!all_chunks_free)
		{
			if (is_from(free_chunk->get(), node.begin(), node.end()))
			{
				std::less<void *> lt;
				void* const end = node.end();
				do
				{
					prev_free_chunk = free_chunk;
					free_chunk = free_chunk->getNext();
				} while (free_chunk && lt(free_chunk->get(), end));
			}

			prev = ptr;
		}
		else
		{
			// All chunks from this block are free
			   // Remove block from list
			if (prev != nullptr)
			{
				prev->setNext(next);
			}
			else
			{
				list = next;
			}
			
			if (prev_free_chunk != nullptr)
			{
				prev_free_chunk->setNext(free_chunk);
			}
			else
			{
				this->first = free_chunk;
			}
			this->addToJunk(saved_free, last_free);

			// And release memory
			(UserAllocator::free)(node.begin());
			node_opool.destroy(ptr);
			ret = true;
		}

		ptr = next;
	}

	next_size = start_size;
	return ret;
}

template <typename UserAllocator>
bool ApraPool<UserAllocator>::is_from(void* const chunk, char* const start, char* const end)
{
	std::less_equal<void *> lt_eq;
	std::less<void *> lt;
	return (lt_eq(start, chunk) && lt(chunk, end));
}

template <typename UserAllocator>
void* ApraPool<UserAllocator>::ordered_malloc(const size_t num_chunks)
{	
	void * ret = store().malloc_n(num_chunks, requested_size);

	if ((ret != 0) || (num_chunks == 0))
	{
		return ret;
	}

	// Not enough memory in our storages; make a new storage,	
	next_size = std::max(next_size, num_chunks);
	size_t POD_size = next_size * requested_size;
	char * ptr = (UserAllocator::malloc)(POD_size);
	if (ptr == 0)
	{
		if (num_chunks < next_size)
		{
			// Try again with just enough memory to do the job, or at least whatever we
			// allocated last time:
			next_size >>= 1;
			next_size = std::max(next_size, num_chunks);
			POD_size = next_size * requested_size;
			ptr = (UserAllocator::malloc)(POD_size);
		}
		if (ptr == nullptr)
		{
			return nullptr;
		}
	}
	
	 ApraNode* node = node_opool.construct(ptr, POD_size);

	// Split up block so we can use what wasn't requested.
	if (next_size > num_chunks)
	{
		store().add_ordered_block(node->begin() + num_chunks * requested_size, POD_size - num_chunks * requested_size, requested_size);
	}

	if (!max_size)
	{
		next_size <<= 1;
	}
	else if (next_size < max_size)
	{
		next_size = std::min(next_size << 1, max_size);
	}

	//  insert it into the list,
	//   handle border case.
	if (list == nullptr || std::greater<void *>()(list->begin(), node->begin()))
	{
		node->setNext(list);
		list = node;
	}
	else
	{
		ApraNode* prev = list;

		while (true)
		{
			// if we're about to hit the end, or if we've found where "node" goes.
			if (prev->getNext() == nullptr || std::greater<void *>()(prev->getNext()->begin(), node->begin()))
			{
				break;
			}

			prev = prev->getNext();
		}

		node->setNext(prev->getNext());
		prev->setNext(node);
	}

	//  and return it.
	return node->begin();
}

template <typename UserAllocator>
void ApraPool<UserAllocator>::ordered_free(void* ptr, const size_t n)
{
	store().ordered_free_n(ptr, n, requested_size);
}