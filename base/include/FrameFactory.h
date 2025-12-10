#pragma once
#include <boost/pool/object_pool.hpp>
#include <boost/shared_ptr.hpp>
#include <mutex>
#include <atomic>

#include "CommonDefs.h"
#include "FrameMetadata.h"
#include "Allocators.h"
#include <memory>

class FrameFactory
{
private:
	boost::object_pool<Frame> frame_allocator;
	std::shared_ptr<HostAllocator> memory_allocator; 
	
	frame_sp eosFrame;
	frame_sp emptyFrame;
	std::mutex m_mutex;

	std::atomic_uint counter;
	std::atomic_size_t numberOfChunks;
	size_t maxConcurrentFrames;
	framemetadata_sp mMetadata;
public:
	FrameFactory(framemetadata_sp metadata, size_t _maxConcurrentFrames=0);
	virtual ~FrameFactory();
	frame_sp create(size_t size, boost::shared_ptr<FrameFactory>& mother);
	// Intended only for command, props, pauseplay 
	// don't use it for normal output frames - Module when sending EOP is using it and some other modules
	frame_sp create(size_t size, boost::shared_ptr<FrameFactory>& mother,framemetadata_sp& metadata);
	frame_sp create(boost::shared_ptr<FrameFactory>& mother);
	frame_sp create(frame_sp &frame, size_t size, boost::shared_ptr<FrameFactory>& mother);	
	void destroy(Frame* pointer);
	frame_sp getEOSFrame() {
		return eosFrame;
	}
	size_t getNumberOfChunks(size_t size);
	framemetadata_sp getFrameMetadata(){return mMetadata;}
	void setMetadata(framemetadata_sp metadata) { mMetadata = metadata;}

	frame_sp getEmptyFrame() { return emptyFrame; }

	std::string getPoolHealthRecord();
};