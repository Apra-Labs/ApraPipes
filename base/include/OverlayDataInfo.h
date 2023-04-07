#include <boost/serialization/vector.hpp>
#include <opencv2/core/types.hpp>

#include "FrameMetadata.h"
#include "Frame.h"
#include "Utils.h"
#include "Module.h"

class RectangleClass;

class OverlayDataInfo
{
public:
	boost::shared_ptr<RectangleClass> rectangle;
};
 
class  RectangleClass : public OverlayDataInfo
{
public:
	RectangleClass()
	{
	}

	void serialize(void* buffer, size_t size)
	{
		auto& info = *this;
		Utils::serialize<OverlayDataInfo>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		return sizeof(OverlayDataInfo)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2);
	}

	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& x1& y1& x2& y2
	}
};