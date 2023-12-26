#include "stdafx.h"
#include "FrameContainerQueue.h"

FrameContainerQueue::FrameContainerQueue(size_t capacity) :bounded_buffer<frame_container>(capacity) {
}

void FrameContainerQueue::push(frame_container item)
{
	bounded_buffer<frame_container>::push(item);
}

void FrameContainerQueue::push_drop_oldest(frame_container item)
{
	bounded_buffer<frame_container>::push_drop_oldest(item);
}

frame_container FrameContainerQueue::pop()
{
	return bounded_buffer<frame_container>::pop();
}

bool FrameContainerQueue::try_push(frame_container item)
{
	return bounded_buffer<frame_container>::try_push(item);
}

bool FrameContainerQueue::push_back(frame_container item) // ws_yash
{
	LOG_DEBUG << "Frame Container Queue PUSH_BACK========================================>>>>>" ;
	return bounded_buffer<frame_container>::push_back(item);
}

frame_container FrameContainerQueue::try_pop()
{
	return bounded_buffer<frame_container>::try_pop();
}

bool FrameContainerQueue::isFull()
{
	return bounded_buffer<frame_container>::isFull();
}

void FrameContainerQueue::clear()
{
	return bounded_buffer<frame_container>::clear();
}

void FrameContainerQueue::flush()
{
	return bounded_buffer<frame_container>::flush();
}

void FrameContainerQueue::accept()
{
	return bounded_buffer<frame_container>::accept();
}

size_t FrameContainerQueue::size()
{
	return bounded_buffer<frame_container>::size();
}
