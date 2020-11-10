#pragma once
#include <boost/pool/object_pool.hpp>
#include <boost/pool/pool.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <atomic>

#ifdef APRA_CUDA_ENABLED
	#include "ApraPool.h"
	#include "apra_cudamallochost_allocator.h"
	#include "apra_cudamalloc_allocator.h"
#endif

#include "CommonDefs.h"
#include "FrameMetadata.h"

class FrameFactory
{
private:
	boost::object_pool<Frame> frame_allocator;
	boost::object_pool<Buffer> buffer_opool;
	boost::pool<> buff_allocator;
#ifdef APRA_CUDA_ENABLED
	boost::pool<apra_cudamallochost_allocator> buff_pinned_allocator;
	ApraPool<apra_cudamalloc_allocator> buff_cudadevice_allocator;
#endif
	frame_sp eosFrame;
	frame_sp emptyFrame;
	boost::mutex m_mutex;

	std::atomic_uint counter;
	std::atomic_size_t numberOfChunks;
	size_t maxConcurrentFrames;
public:
	FrameFactory(size_t _maxConcurrentFrames=0);
	virtual ~FrameFactory();
	frame_sp create(size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType);
	buffer_sp createBuffer(size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType);
	frame_sp create(buffer_sp buffer, size_t size, boost::shared_ptr<FrameFactory>& mother, FrameMetadata::MemType memType);
	void destroy(Frame* pointer, FrameMetadata::MemType memType);
	void destroyBuffer(Buffer* pointer, FrameMetadata::MemType memType);
	frame_sp getEOSFrame() {
		return eosFrame;
	}

	frame_sp getEmptyFrame() { return emptyFrame; }

	std::string getPoolHealthRecord();
};
