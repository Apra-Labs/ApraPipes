#pragma once

#include <opencv2/core/types_c.h>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

/***
 * ApraPoint3f implements the serialize methods
 *
*/

class ApraPoint3f : public cv::Point3f
{
public:
	ApraPoint3f() {}

	ApraPoint3f(float _x, float _y, float _z) : cv::Point3f(_x, _y, _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	ApraPoint3f(cv::Point3f& point) : cv::Point3f(point)
	{

	}
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& x;
		ar& y;
		ar& z;
	}
};