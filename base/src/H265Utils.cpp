#include "stdafx.h"
#include "H265Utils.h"

H265Utils::H265_NAL_TYPE H265Utils::getNALUType(const char* buffer)
{
	return (H265_NAL_TYPE)((buffer[4] >> 1) & 0x3F);
}

bool H265Utils::isIDR(H265_NAL_TYPE type)
{
	return (type == IDR_W_RADL || type == IDR_N_LP);
}