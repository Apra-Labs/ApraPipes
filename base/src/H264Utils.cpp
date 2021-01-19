#include "stdafx.h"
#include "H264Utils.h"
#include "Frame.h"




H264Utils::H264_NAL_TYPE H264Utils::getNALUType(Frame *frm)
{
	char* p1 = static_cast<char*>(frm->data());
	return getNALUType(p1);
}

H264Utils::H264_NAL_TYPE H264Utils::getNALUType(const char *buffer)
{
	return (H264_NAL_TYPE)(buffer[4] & 0x1F);
}



bool H264Utils::getNALUnit(const char *buffer, size_t length, size_t &offset)
{
	if (length < 3) return false;
	size_t cnt = 3;

	while (cnt < length)
	{
		if (buffer[cnt - 1] == 0x1 && buffer[cnt - 2] == 0x0 && buffer[cnt - 3] == 0x0)
		{
			offset = cnt;
			return true;
		}
		cnt++;
	}

	return false;
}