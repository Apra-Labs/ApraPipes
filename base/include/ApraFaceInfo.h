#pragma once

#include <opencv2/core/types_c.h>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

class ApraFaceInfo
{
public:
	float x1, x2, y1, y2, score;

	ApraFaceInfo(): x1(0), y1(0), x2(0), y2(0), score(0)
	{

	}

	size_t getSerializeSize()
	{
		return sizeof(x1) + sizeof(x2) + sizeof(y1) + sizeof(y2) + sizeof(score);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &x1 &x2 &y1 &y2 &score;
	}
};