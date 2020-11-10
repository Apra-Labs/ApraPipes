#pragma once
#include <string>
#include "opencv2/opencv.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

class RawImageMetadata;

class Utils {
private:
	Utils(void) {}
public:
	static int64_t GetEpocFromTime(const char * t);
	static int64_t GetEpocFromTimeInMillis(const char * t);
	static std::string base64_encode(unsigned char const* bytes_to_encode, size_t in_len);
	static cv::Mat getMatHeader(RawImageMetadata* metadata);
	static cv::Mat getMatHeader(cv::Rect& roi, RawImageMetadata* metadata);
	static cv::cuda::GpuMat getGPUMatHeader(cv::Rect& roi, RawImageMetadata* metadata);
	static cv::Mat getMatHeader(int width, int height, int type);
	static void round_roi(cv::Rect& roi, int alignLength);
	static bool check_roi_bounds(cv::Rect& roi, int width, int height);

	template<class T>
	static void serialize(T& obj, void* buffer, size_t size)
	{
		boost::iostreams::basic_array_sink<char> device_sink((char*)buffer, size);
		boost::iostreams::stream<boost::iostreams::basic_array_sink<char> > s_sink(device_sink);

		boost::archive::binary_oarchive oa(s_sink);
		oa << obj;
	}

	template<class T>
	static void deSerialize(T& obj, void* buffer, size_t size)
	{
		boost::iostreams::basic_array_source<char> device((char*)buffer, size);
		boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
		boost::archive::binary_iarchive ia(s);

		ia >> obj;
	}
};