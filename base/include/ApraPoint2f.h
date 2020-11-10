#pragma once

#include <opencv2/core/types_c.h>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

/***
 * cv::Point serialize is not supported
 * ApraPoint2f implements the serialize methods
 * 
*/

class ApraPoint2f : public cv::Point2f
{
public:
	ApraPoint2f() {}

	ApraPoint2f(cv::Point2f point) : cv::Point2f(point)
	{

	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & x;
		ar & y;
	}
};