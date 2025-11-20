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

std::optional<ApraChunk*> ApraSegregatedStorage::getFreeApraChunk()
{
	if (junkList == nullptr)
	{
		junkList = chunk_opool.construct();
	}

	ApraChunk* const ret = junkList;
	junkList = junkList->getNext().value_or(nullptr);

	ret->set(nullptr, nullptr);

	return ret;
}

bool ApraSegregatedStorage::empty() const
{
	return first == nullptr;
}

void ApraSegregatedStorage::add_ordered_block(void * const block, const size_t nsz, const size_t npartition_sz)
{
	auto loc = find_prev(block);

	// Place either at beginning or in middle/end
	if (!loc)
	{
		first = segregate(block, nsz, npartition_sz, first);
	}
	else
	{
		(*loc)->setNext(segregate(block, nsz, npartition_sz, (*loc)->getNext().value_or(nullptr)));
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
	auto chunk = *getFreeApraChunk();
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
		auto temp = *getFreeApraChunk();
		temp->set(iter, chunk);
		chunk = temp;
	}

	// Point the first pointer, too
	auto temp = *getFreeApraChunk();
	temp->set(block, chunk);

	return temp;
}

std::optional<ApraChunk*> ApraSegregatedStorage::find_prev(void * const ptr)
{
	// Handle border case.
	if (first == nullptr || std::greater<void *>()(first->get(), ptr))
	{
		return std::nullopt;
	}

	ApraChunk* iter = first;
	while (true)
	{
		// if we're about to hit the end, or if we've found where "ptr" goes.
		auto iterNext = iter->getNext();
		if (!iterNext || std::greater<void *>()((*iterNext)->get(), ptr))
			return iter;

		iter = *iterNext;
	}
}

ApraChunk* ApraSegregatedStorage::try_malloc_n(ApraChunk*& start, size_t n, const size_t requested_size)
{
	ApraChunk* iter = start->getNext().value_or(nullptr);
	while (--n != 0)
	{
		auto nextOpt = iter->getNext();
		ApraChunk* next = nextOpt.value_or(nullptr);
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
		auto startNext = start->getNext();
		if (start == nullptr || !startNext)
		{
			return nullptr;
		}

		iter = try_malloc_n(start, n, requested_size);
	} while (iter == nullptr);

	void* ret = start->getNext().value_or(nullptr)->get();
	if (start->get() == nullptr)
	{
		first = iter->getNext().value_or(nullptr);
		addToJunk(start->getNext().value_or(nullptr), iter);
	}
	else
	{
		auto nextFree = iter->getNext().value_or(nullptr);
		addToJunk(start->getNext().value_or(nullptr), iter);
		start->setNext(nextFree);
	}	

	return ret;
}
