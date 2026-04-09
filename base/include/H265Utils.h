#pragma once

#include <stddef.h>
#include <boost/asio/buffer.hpp>
#include <tuple>

using namespace boost::asio;

class H265Utils {
private:
	H265Utils(void) {}
public:
	enum H265_NAL_TYPE {
		IDR_W_RADL = 19,
		IDR_N_LP = 20,
		VPS = 32,
		SPS = 33,
		PPS = 34,
		SEI_PREFIX = 39
	};

	static H265_NAL_TYPE getNALUType(const char *buffer);
	static bool isIDR(H265_NAL_TYPE type);
	static bool getNALUnit(const char *buffer, size_t length, size_t &offset);
	static std::tuple<short, const_buffer, const_buffer, const_buffer> parseNalu(const const_buffer input);
};