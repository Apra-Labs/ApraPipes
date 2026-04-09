#pragma once

#include <stddef.h>

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
};