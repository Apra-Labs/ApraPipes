#pragma once

#include "Frame.h"

class ExtFrame : public Frame
{
public:
	ExtFrame(void* data, size_t size) : Frame()
	{
		mData = data;
		mSize = size;
	}

	virtual ~ExtFrame()
	{
		
	}

	void* data() const BOOST_ASIO_NOEXCEPT
	{
		return mData;
	}

	std::size_t size() const BOOST_ASIO_NOEXCEPT
	{
		return mSize;
	}

private:
	void* mData;
	size_t mSize;
};