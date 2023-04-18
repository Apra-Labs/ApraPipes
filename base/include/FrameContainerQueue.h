#pragma once

#include <boost/shared_ptr.hpp>
#include "BoundBuffer.h"
#include "CommonDefs.h"
#include "Overlay.h"
#include "Module.h"
#include "Frame.h"
#include "FaceDetectsInfo.h"

class FrameContainerQueue :public bounded_buffer<frame_container> {
public:
	FrameContainerQueue(size_t capacity);
	virtual void push(frame_container item);
	virtual void push_drop_oldest(frame_container item);
	virtual frame_container pop();

	virtual bool try_push(frame_container item);
	virtual frame_container try_pop();

	virtual bool isFull();
	virtual void clear();
	virtual void flush();
	virtual void accept();
	virtual size_t size();

private:
};

class FrameContainerQueueAdapter : public FrameContainerQueue
{
public:
	FrameContainerQueueAdapter() : FrameContainerQueue(0) {}
	void adapt(boost::shared_ptr<FrameContainerQueue> adaptee) {
		mAdaptee = adaptee;
	}
	void push(frame_container item) {
		if (mAdaptee.get() != nullptr)
		{
			PushType p = should_push(item);
			if (p == MUST_PUSH)
			{
				mAdaptee->push(item);
			}
			else if (p == TRY_PUSH)
			{
				if (mAdaptee->try_push(item))
				{
					on_push_success(item);
				}
				else
				{
					on_failed_push(item);
				}
			}
		}
	}
	frame_container pop() {
		if (mAdaptee.get() != nullptr)
		{
			frame_container ret = mAdaptee->try_pop();
			if (ret.size() == 0)
			{
				return on_failed_pop();
			}
			else {
				return on_pop_success(ret);
			}
		}
		return frame_container();
	}
	bool try_push(frame_container item)
	{
		if (mAdaptee.get() != nullptr)
		{
			PushType p = should_push(item);
			if (p == DONT_PUSH)
			{
				//dont call on_failed_push here
				return false;
			}
			else if (p == MUST_PUSH)
			{
				mAdaptee->push(item);
				on_push_success(item);
				return true;
			}
			else //if (p == TRY_PUSH)
			{
				if (mAdaptee->try_push(item))
				{
					on_push_success(item);
					return true;
				}
			}
		}
		return false;
	}
	frame_container try_pop()
	{
		if (mAdaptee.get() != nullptr)
		{
			return mAdaptee->try_pop();
		}
		return frame_container();
	}

	bool isFull()
	{
		if(!mAdaptee.get())
		{
			return false;
		}

		return mAdaptee->isFull();
	}

	void clear()
	{
		if(mAdaptee.get())
		{
			mAdaptee->clear();
		}
	}

	void accept()
	{
		if(mAdaptee.get())
		{
			mAdaptee->accept();
		}
	}

	size_t size()
	{
		if(!mAdaptee.get())
		{
			return 0;
		}

		return mAdaptee->size();
	}

protected:
	boost::shared_ptr<FrameContainerQueue> mAdaptee;
	enum PushType { DONT_PUSH = 0, TRY_PUSH = 1, MUST_PUSH = 2 };
	virtual PushType should_push(frame_container item) { return TRY_PUSH; }
	virtual void on_failed_push(frame_container item) {}
	virtual void on_push_success(frame_container item) {}
	virtual frame_container on_failed_pop() {
		return frame_container();
	}
	virtual frame_container  on_pop_success(frame_container item) {
		return item;
	}
};

// class FrameContainerQueueOverlayAdapter : public FrameContainerQueue
// {
// public:
// 	FrameContainerQueueOverlayAdapter(std::function<frame_sp(size_t)> _makeFrame) : FrameContainerQueue(0) {
// 		makeFrame = _makeFrame;
// 	}
// 	void adapt(boost::shared_ptr<FrameContainerQueue> adaptee) {
// 		mAdaptee = adaptee;
// 	}
// 	void push(frame_container item) {
// 		if (mAdaptee.get() != nullptr)
// 		{
// 			for (auto it = item.begin(); it != item.cend(); it++)
// 			{
// 				auto frameType = it->second->mFrameType;
// 				if (frameType == FrameMetadata::FACEDETECTS_INFO)
// 				{
// 					mOverlayInfo.reset(new RectangleOverlay());
// 					FaceDetectsInfo result = FaceDetectsInfo::deSerialize(item);
// 					auto overlayFrame = makeFrame(mOverlayInfo->getSerializeSize());
// 				}
// 			}
// 		}
// 	}
// 	frame_container pop() {
// 		if (mAdaptee.get() != nullptr)
// 		{
// 			frame_container ret = mAdaptee->try_pop();
// 			if (ret.size() == 0)
// 			{
// 				return on_failed_pop();
// 			}
// 			else {
// 				return on_pop_success(ret);
// 			}
// 		}
// 		return frame_container();
// 	}
// 	bool try_push(frame_container item)
// 	{
// 		if (mAdaptee.get() != nullptr)
// 		{
			
// 		}
// 		return false;
// 	}
// 	frame_container try_pop()
// 	{
// 		if (mAdaptee.get() != nullptr)
// 		{
// 			return mAdaptee->try_pop();
// 		}
// 		return frame_container();
// 	}

// 	bool isFull()
// 	{
// 		if (!mAdaptee.get())
// 		{
// 			return false;
// 		}

// 		return mAdaptee->isFull();
// 	}

// 	void clear()
// 	{
// 		if (mAdaptee.get())
// 		{
// 			mAdaptee->clear();
// 		}
// 	}

// 	void accept()
// 	{
// 		if (mAdaptee.get())
// 		{
// 			mAdaptee->accept();
// 		}
// 	}

// 	size_t size()
// 	{
// 		if (!mAdaptee.get())
// 		{
// 			return 0;
// 		}

// 		return mAdaptee->size();
// 	}

// protected:
// 	boost::shared_ptr<FrameContainerQueue> mAdaptee;
// 	virtual frame_container on_failed_pop() {
// 		return frame_container();
// 	}
// 	virtual frame_container  on_pop_success(frame_container item) {
// 		return item;
// 	}
// 	boost::shared_ptr<OverlayDataInfo> mOverlayInfo;
// 	std::function<frame_sp(size_t)> makeFrame
// };