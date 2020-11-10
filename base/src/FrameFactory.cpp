#include "stdafx.h"
#include <boost/bind.hpp>
#include "FrameFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"

#define LOG_FRAME_FACTORY

#define CHUNK_SZ 1024
FrameFactory::FrameFactory(size_t _maxConcurrentFrames) :
	  maxConcurrentFrames(_maxConcurrentFrames)
	, buff_allocator(CHUNK_SZ)
#ifdef APRA_CUDA_ENABLED
	, buff_pinned_allocator(CHUNK_SZ)
	, buff_cudadevice_allocator(CHUNK_SZ)
#endif
{
	eosFrame = frame_sp(new EoSFrame());
	emptyFrame = frame_sp(new EmptyFrame());
	counter = 0;
	numberOfChunks = 0;
}
FrameFactory::~FrameFactory() {
	buff_allocator.release_memory();
#ifdef APRA_CUDA_ENABLED
	buff_pinned_allocator.release_memory();
	buff_cudadevice_allocator.release_memory();
#endif
}

size_t getNumberOfChunks(size_t size)
{
	return (size + CHUNK_SZ - 1) / CHUNK_SZ;
}

frame_sp FrameFactory::create(size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType)
{
	boost::mutex::scoped_lock lock(m_mutex);
	if (maxConcurrentFrames && counter >= maxConcurrentFrames)
	{
		return frame_sp();
	}
	size_t n = getNumberOfChunks(size);

	counter.fetch_add(1, memory_order_seq_cst);
	numberOfChunks.fetch_add(n, memory_order_seq_cst);		
	
#ifdef APRA_CUDA_ENABLED
	if (memType == FrameMetadata::MemType::HOST_PINNED)
	{
		return frame_sp(
			frame_allocator.construct(buff_pinned_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroy, this, _1, memType));
	}
	else if (memType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		return frame_sp(
			frame_allocator.construct(buff_cudadevice_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroy, this, _1, memType));
	}
	else
#endif	
		return frame_sp(
			frame_allocator.construct(buff_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroy, this, _1, memType));
}

buffer_sp FrameFactory::createBuffer(size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType)
{
	boost::mutex::scoped_lock lock(m_mutex);
	if (maxConcurrentFrames && counter >= maxConcurrentFrames)
	{
		return buffer_sp();
	}
	size_t n = getNumberOfChunks(size);

	counter.fetch_add(1, memory_order_seq_cst);
	numberOfChunks.fetch_add(n, memory_order_seq_cst);

#ifdef APRA_CUDA_ENABLED
	if (memType == FrameMetadata::MemType::HOST_PINNED)
	{
		return buffer_sp(
			buffer_opool.construct(buff_pinned_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroyBuffer, this, _1, memType));
	}
	else if (memType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		return buffer_sp(
			buffer_opool.construct(buff_cudadevice_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroyBuffer, this, _1, memType));
	}
	else
#endif	
		return buffer_sp(
			buffer_opool.construct(buff_allocator.ordered_malloc(n), size, mother),
			boost::bind(&FrameFactory::destroyBuffer, this, _1, memType));
	
}

void FrameFactory::destroy(Frame* pointer, FrameMetadata::MemType memType)
{				
	boost::mutex::scoped_lock lock(m_mutex);
	counter.fetch_sub(1, memory_order_seq_cst);

	if (pointer->myOrig != NULL)
	{
		size_t n = getNumberOfChunks(pointer->size());
		numberOfChunks.fetch_sub(n, memory_order_seq_cst);
#ifdef APRA_CUDA_ENABLED
		if (memType == FrameMetadata::MemType::HOST_PINNED)
		{
			buff_pinned_allocator.ordered_free(pointer->myOrig, n);
		}
		else if (memType == FrameMetadata::MemType::CUDA_DEVICE)
		{
			buff_cudadevice_allocator.ordered_free(pointer->myOrig, n);
		}
		else
#endif
		buff_allocator.ordered_free(pointer->myOrig, n);
	}

	auto mother = pointer->myMother;
	pointer->~Frame();
	frame_allocator.free(pointer);
}

void FrameFactory::destroyBuffer(Buffer* pointer, FrameMetadata::MemType memType)
{		
	boost::mutex::scoped_lock lock(m_mutex);
	counter.fetch_sub(1, memory_order_seq_cst);

	if (pointer->myOrig != NULL)
	{
		size_t n = getNumberOfChunks(pointer->size());
		numberOfChunks.fetch_sub(n, memory_order_seq_cst);
		#ifdef APRA_CUDA_ENABLED
		if (memType == FrameMetadata::MemType::HOST_PINNED)
		{
			buff_pinned_allocator.ordered_free(pointer->myOrig, n);
		}
		else if (memType == FrameMetadata::MemType::CUDA_DEVICE)
		{
			buff_cudadevice_allocator.ordered_free(pointer->myOrig, n);
		}
		else
#endif
		buff_allocator.ordered_free(pointer->myOrig, n);
	}

	auto mother = pointer->myMother;
	pointer->~Buffer();
	buffer_opool.free(pointer);
}

frame_sp FrameFactory::create(buffer_sp buffer, size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType)
{
	size_t oldChunks = getNumberOfChunks(buffer->size());
	size_t newChunks = getNumberOfChunks(size);
	size_t chunksToFree = oldChunks - newChunks;

	auto origPtr = buffer->myOrig;
	if (origPtr == NULL)
	{
		throw AIPException(AIP_FATAL, string("oldFrame->myOrig in NULL. Not expected."));
	}

	if (chunksToFree < 0)
	{
		throw AIPException(AIP_NOTIMPLEMENTED, string("increasing chunks not yet implemented"));
	}
			
	boost::mutex::scoped_lock lock(m_mutex);
	counter.fetch_add(1, memory_order_seq_cst);

	if (chunksToFree >  0)
	{
		numberOfChunks.fetch_sub(chunksToFree, memory_order_seq_cst);
		auto ptr = (void*)((char*)origPtr + (newChunks * CHUNK_SZ));	
#ifdef APRA_CUDA_ENABLED
		if (memType == FrameMetadata::MemType::HOST_PINNED)
		{			
			buff_pinned_allocator.ordered_free(ptr, chunksToFree);
		}
		else if (memType == FrameMetadata::MemType::CUDA_DEVICE)
		{
			buff_cudadevice_allocator.ordered_free(ptr, chunksToFree);
		}
		else
#endif
			buff_allocator.ordered_free(ptr, chunksToFree);
	}
		
	buffer->resetMemory(); // so that when destroyBuffer is called it should not free the memory

	return frame_sp(
		frame_allocator.construct(origPtr, size, mother),
		boost::bind(&FrameFactory::destroy, this, _1, memType));				
}

std::string FrameFactory::getPoolHealthRecord()
{
	std::ostringstream stream;
	stream << "Chunks<" << numberOfChunks << "> TotalBytes<" <<  numberOfChunks * CHUNK_SZ << "> Frames<" << counter << ">";
	
	return stream.str();
}

