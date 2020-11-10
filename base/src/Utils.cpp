#include "stdafx.h"
#include "Utils.h"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>
#include <locale>

#include "RawImageMetadata.h"

int64_t Utils::GetEpocFromTime(const char * inp) {
	int64_t x = 0;
	std::string t(inp);
	if (t.length() != 0) {
		try {
			boost::posix_time::ptime abs_time;// = boost::posix_time::time_from_string(t.c_str());
			boost::posix_time::time_input_facet *tif = new boost::posix_time::time_input_facet;
			tif->set_iso_extended_format();
			std::istringstream iss(t);
			iss.imbue(std::locale(std::locale::classic(), tif));
			iss >> abs_time;
			boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
			x = (abs_time - epoch).total_seconds();
			
		}
		catch (std::exception &ex) {
			LOG_ERROR << "exception occured in parsing sps buffer:" <<ex.what();
		}
	}
	return x;
}

int64_t Utils::GetEpocFromTimeInMillis(const char * inp) {
	int64_t x = 0;
	std::string t(inp);
	if (t.length() != 0) {
		try {
			boost::posix_time::ptime abs_time;// = boost::posix_time::time_from_string(t.c_str());
			boost::posix_time::time_input_facet *tif = new boost::posix_time::time_input_facet;
			tif->set_iso_extended_format();
			std::istringstream iss(t);
			iss.imbue(std::locale(std::locale::classic(), tif));
			iss >> abs_time;
			boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
			x = (abs_time - epoch).total_milliseconds();

		}
		catch (std::exception &ex) {
			LOG_ERROR << "exception occured in parsing sps buffer:" << ex.what();
		}
	}
	return x;
}

std::string Utils::base64_encode(unsigned char const* bytes_to_encode, size_t in_len) {

	static const std::string base64_chars =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz"
		"0123456789+/";

	std::string ret;
	int i = 0;
	int j = 0;
	unsigned char char_array_3[3];
	unsigned char char_array_4[4];

	while (in_len--) {
		char_array_3[i++] = *(bytes_to_encode++);
		if (i == 3) {
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;

			for (i = 0; (i < 4); i++)
				ret += base64_chars[char_array_4[i]];
			i = 0;
		}
	}

	if (i)
	{
		for (j = i; j < 3; j++)
			char_array_3[j] = '\0';

		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

		for (j = 0; (j < i + 1); j++)
			ret += base64_chars[char_array_4[j]];

		while ((i++ < 3))
			ret += '=';

	}

	return ret;

}

cv::Mat Utils::getMatHeader(RawImageMetadata* metadata)
{
	uint8_t data;	
	return cv::Mat(metadata->getHeight(), metadata->getWidth(), metadata->getType(), static_cast<void*>(&data), metadata->getStep());
}

cv::Mat Utils::getMatHeader(cv::Rect& roi, RawImageMetadata* metadata)
{
	uint8_t data;
	return cv::Mat(roi.height, roi.width, metadata->getType(), static_cast<void*>(&data), metadata->getStep());
}

cv::Mat Utils::getMatHeader(int width, int height, int type)
{
	uint8_t data;	
	return cv::Mat(height, width, type, static_cast<void*>(&data));
}

cv::cuda::GpuMat Utils::getGPUMatHeader(cv::Rect& roi, RawImageMetadata* rawMetadata)
{
	uint8_t data;
    return cv::cuda::GpuMat(roi.height, roi.width, rawMetadata->getType(),  static_cast<void*>(&data), rawMetadata->getStep());
}

void Utils::round_roi(cv::Rect &roi, int alignLength)
{
	auto extra = roi.x % alignLength;
	roi.x = roi.x - extra;
	roi.width += extra;

	extra = roi.y % alignLength;
	roi.y = roi.y - extra;
	roi.height += extra;
}

bool Utils::check_roi_bounds(cv::Rect& roi, int width, int height)
{
	if(roi.width <= 0)
	{
		roi.width = width;
	}

	if(roi.height <= 0)
	{
		roi.height = height;
	}

	if (roi.x < 0 || roi.y < 0 || (roi.x + roi.width) > width || (roi.y + roi.height) > height)
	{
		// resetting the roi
		roi.x = 0;
		roi.y = 0;
		roi.width = width;
		roi.height = height;

		return false;
	}

	return true;
}