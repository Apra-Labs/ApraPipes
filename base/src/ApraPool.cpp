#include "ApraPool.h"

ApraSegregatedStorage::ApraSegregatedStorage() :first(nullptr), junkList(nullptr)
{

}

ApraSegregatedStorage::~ApraSegregatedStorage()
{
	releaseChunks();	
}

void ApraSegregatedStorage::releaseChunks()
{		
	first = nullptr; 
}

void ApraSegregatedStorage::addToJunk(ApraChunk* start, ApraChunk* end)
{
	end->setNext(junkList);
	junkList = start;
}

ApraChunk* ApraSegregatedStorage::getFreeApraChunk()
{
	if (junkList == nullptr)
	{
		junkList = chunk_opool.construct();
	}

	ApraChunk* const ret = junkList;
	junkList = junkList->getNext();

	ret->set(nullptr, nullptr);

	return ret;
}

bool ApraSegregatedStorage::empty() const
{
	return first == nullptr;
}

void ApraSegregatedStorage::add_ordered_block(void * const block, const size_t nsz, const size_t npartition_sz)
{
	ApraChunk* const loc = find_prev(block);

	// Place either at beginning or in middle/end
	if (loc == 0)
	{
		first = segregate(block, nsz, npartition_sz, first);
	}
	else
	{
		loc->setNext(segregate(block, nsz, npartition_sz, loc->getNext()));
	}
}

void ApraSegregatedStorage::ordered_free_n(void * const chunks, const size_t n, const size_t requested_size)
{
	if (n != 0)
	{
		add_ordered_block(chunks, n * requested_size, requested_size);
	}
}

ApraChunk* ApraSegregatedStorage::segregate(void* block, size_t sz, size_t partition_sz, ApraChunk* end)
{

	char* old = static_cast<char*>(block) + ((sz - partition_sz) / partition_sz) * partition_sz;

	// Set it to point to the end
	auto chunk = getFreeApraChunk();
	chunk->set(old, end);

	// Handle border case where sz == partition_sz (i.e., we're handling an array
	//  of 1 element)
	if (old == block)
	{
		return chunk;
	}

	// Iterate backwards, building a singly-linked list of pointers
	for (char * iter = old - partition_sz; iter != block; old = iter, iter -= partition_sz)
	{
		auto temp = getFreeApraChunk();
		temp->set(iter, chunk);
		chunk = temp;
	}

	// Point the first pointer, too
	auto temp = getFreeApraChunk();
	temp->set(block, chunk);

	return temp;
}

ApraChunk* ApraSegregatedStorage::find_prev(void * const ptr)
{
	// Handle border case.
	if (first == nullptr || std::greater<void *>()(first->get(), ptr))
	{
		return nullptr;
	}

	ApraChunk* iter = first;
	while (true)
	{
		// if we're about to hit the end, or if we've found where "ptr" goes.
		if (iter->getNext() == nullptr || std::greater<void *>()(iter->getNext()->get(), ptr))
			return iter;

		iter = iter->getNext();
	}
}

ApraChunk* ApraSegregatedStorage::try_malloc_n(ApraChunk*& start, size_t n, const size_t requested_size)
{
	ApraChunk* iter = start->getNext();
	while (--n != 0)
	{
		ApraChunk* next = iter->getNext();
		if (next == nullptr || next->get() != static_cast<char *>(iter->get()) + requested_size)
		{
			start = iter;
			return nullptr;
		}
		iter = next;
	}
	return iter;
}

void* ApraSegregatedStorage::malloc_n(const size_t n, const size_t requested_size)
{
	if (n == 0)
	{
		return nullptr;
	}

	ApraChunk chunk;
	chunk.set(nullptr, first);

	ApraChunk* start = &chunk;
	ApraChunk* iter;
	do
	{
		if (start == nullptr || start->getNext() == nullptr)
		{
			return nullptr;
		}

		iter = try_malloc_n(start, n, requested_size);
	} while (iter == nullptr);
	
	void* ret = start->getNext()->get();	
	if (start->get() == nullptr)
	{		
		first = iter->getNext();
		addToJunk(start->getNext(), iter);
	}
	else
	{
		auto nextFree = iter->getNext();
		addToJunk(start->getNext(), iter);
		start->setNext(nextFree);
	}	

	return ret;
}
