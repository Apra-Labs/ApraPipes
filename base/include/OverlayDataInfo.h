#pragma once

#include <boost/serialization/vector.hpp>
#include "Utils.h"
#include "Module.h"


class OverlayInfo
{
public:
	OverlayInfo() {}
	virtual void serialize(void* buffer, size_t size) {}
	virtual size_t getSerializeSize() = 0;
};
 
class  RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() {}

	void serialize(void* buffer, size_t size)
	{
		auto& info = *this;
		Utils::serialize<RectangleOverlay>(info, buffer, size);
	}

	static RectangleOverlay deSerialize(frame_sp& frame)
	{
		RectangleOverlay result;
		
		Utils::deSerialize<RectangleOverlay>(result, frame->data(), frame->size());

		return result;
	}

	size_t getSerializeSize()
	{
		return sizeof(RectangleOverlay)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + 32;
	}

	float x1, y1, x2, y2;
	std::vector<RectangleOverlay> vectorRect;
private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& x1& y1& x2& y2;
	}
};