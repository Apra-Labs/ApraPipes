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

// typefound, iFrame/PFrame, sps(optional),pps (optional)
std::tuple<short, const_buffer, const_buffer, const_buffer> H264Utils::parseNalu(mutable_buffer& input) 
{
	short typeFound = 0;
	char* p1 = static_cast<char*>(input.data());
	size_t offset = 0;
	typeFound = getNALUType(p1);

	if (typeFound == H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
	{
		return { typeFound, input, const_buffer(), const_buffer() };
	}

	if (typeFound == H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM)
	{
		size_t offset = 0;

		if (getNALUnit(p1, input.size(), offset)) // where does it start
		{
			input += offset;
			offset = 0;
			p1 = static_cast<char*>(input.data());

			if (getNALUnit(p1, input.size(), offset)) // where does it end
			{
				char* spsBits = p1;
				// we see 0 0 0 1 as well as 0 0 1
				size_t nSize = offset - 3;
				if (p1[offset - 4] == 0x00)
					nSize--;
				size_t spsSize = nSize;
				auto spsBuffer = const_buffer(spsBits, spsSize);
				input += offset;
				p1 = static_cast<char*>(input.data());

				if (getNALUnit(p1, input.size(), offset))
				{
					char* ppsBits = p1;
					size_t nSize = offset - 3;
					if (p1[offset - 4] == 0x00)
						nSize--;
					size_t ppsSize = nSize;
					auto ppsBuffer = const_buffer(ppsBits, ppsSize);
					// since we are here lets find the next type
					typeFound = getNALUType(p1 + offset - 4); // always looks at 5th byte
					input += offset - 4;
					p1 = static_cast<char*>(input.data());
					auto frameSize = static_cast<size_t>(input.size());
					auto frame = const_buffer(p1, frameSize);
					return { typeFound, frame, spsBuffer, ppsBuffer };
				}
			}
		}
	}
	typeFound = getNALUType(p1 + offset - 4);
	return { typeFound, input, const_buffer(), const_buffer() };
}