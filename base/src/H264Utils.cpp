#include "stdafx.h"
#include "H264Utils.h"
#include "Frame.h"

H264Utils::H264_NAL_TYPE H264Utils::getNALUType(Frame* frm)
{
	char* p1 = static_cast<char*>(frm->data());
	return getNALUType(p1);
}

H264Utils::H264_NAL_TYPE H264Utils::getNALUType(const char* buffer)
{
	return (H264_NAL_TYPE)(buffer[4] & 0x1F);
}

bool H264Utils::getNALUnit(const char* buffer, size_t length, size_t& offset)
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
std::tuple<short, const_buffer, const_buffer> H264Utils::parseNalu(const const_buffer input)
{
	short typeFound = 0;
	char* p1 = reinterpret_cast<char*>(const_cast<void*>(input.data()));
	size_t offset = 0;
	typeFound = getNALUType(p1);

	if (typeFound == H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
	{
		return { typeFound, const_buffer(), const_buffer() };
	}

	if (typeFound == H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM)
	{
		size_t offset = 0;

		if (getNALUnit(p1, input.size(), offset)) // where does it start
		{
			p1 = p1 + offset;
			offset = 0;

			if (getNALUnit(p1, input.size(), offset)) // where does it end
			{
				char* spsBits = p1;
				// we see 0 0 0 1 as well as 0 0 1
				size_t nSize = offset - 3;
				if (p1[offset - 4] == 0x00)
					nSize--;
				size_t spsSize = nSize;
				auto spsBuffer = const_buffer(spsBits, spsSize);
				p1 = p1 + offset;

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
					p1 = p1 + offset - 4;

					auto frameSize = static_cast<size_t>(input.size());
					auto frame = const_buffer(p1, frameSize);
					return { typeFound, spsBuffer, ppsBuffer };
				}
			}
		}
	}
	typeFound = getNALUType(p1 + offset - 4);
	return { typeFound, const_buffer(), const_buffer() };
}

bool H264Utils::getNALUnitOffsetAndSizeBasedOnGivenType(char* buffer, size_t length, size_t& offset, int& naluSeparatorSize, H264Utils::H264_NAL_TYPE naluType, bool checkByType)
{
	if (length < 3) return false;
	size_t cnt = 5;

	while (cnt < length)
	{
		if (buffer[cnt - 3] == 0x1 && buffer[cnt - 4] == 0x0 && buffer[cnt - 5] == 0x0)
		{
			char type = (buffer[cnt - 2] & 0x1F);
			if (type == naluType || !checkByType)
			{
				naluSeparatorSize = 3;
				offset = cnt - 2;
				return true;
			}
		}
		else if (buffer[cnt - 2] == 0x1 && buffer[cnt - 3] == 0x0 && buffer[cnt - 4] == 0x0 && buffer[cnt - 5] == 0x0)
		{
			char type = (buffer[cnt - 1] & 0x1F);
			if (type == naluType || !checkByType)
			{
				naluSeparatorSize = 4;
				offset = cnt - 1;
				return true;
			}
		}
		cnt++;
	}

	return false;
}

bool H264Utils::extractSpsAndPpsFromExtradata(char* buffer, size_t length, char*& sps, int& spsSize, char*& pps, int& ppsSize)
{
	if (length < 3) return false;
	size_t cnt = 5;
	spsSize = 0;
	ppsSize = 0;
	while ((cnt < length) && (!spsSize || !ppsSize))
	{
		if (buffer[cnt - 3] == 0x1 && buffer[cnt - 4] == 0x0 && buffer[cnt - 5] == 0x0)
		{
			char type = (buffer[cnt - 2] & 0x1F);
			if (type == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				sps = buffer + (cnt - 5);
				size_t offset = 0;
				int naluSeparatorSize = 0;
				getNALUnitOffsetAndSizeBasedOnGivenType(buffer + (cnt - 2), length - (cnt - 2), offset, naluSeparatorSize, H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_NON_IDR_SLICE, false);
				spsSize = offset + 3 - naluSeparatorSize;
				cnt += offset - naluSeparatorSize;
			}
			if (type == H264Utils::H264_NAL_TYPE_PIC_PARAM)
			{
				pps = buffer + (cnt - 5);
				size_t offset = 0;
				int naluSeparatorSize = 0;
				getNALUnitOffsetAndSizeBasedOnGivenType(buffer + (cnt - 2), length - (cnt - 2), offset, naluSeparatorSize, H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_NON_IDR_SLICE, false);
				if (!offset && !naluSeparatorSize)
				{
					ppsSize = length - (cnt - 5);
				}
				else
				{
					ppsSize = offset + 3 - naluSeparatorSize;
				}
				cnt += offset - naluSeparatorSize;
			}
		}
		else if (buffer[cnt - 2] == 0x1 && buffer[cnt - 3] == 0x0 && buffer[cnt - 4] == 0x0 && buffer[cnt - 5] == 0x0)
		{
			char type = (buffer[cnt - 1] & 0x1F);
			if (type == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				sps = buffer + (cnt - 5);
				size_t offset = 0;
				int naluSeparatorSize = 0;
				getNALUnitOffsetAndSizeBasedOnGivenType(buffer + (cnt - 1), length - (cnt - 1), offset, naluSeparatorSize, H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_NON_IDR_SLICE, false);
				spsSize = offset + 4 - naluSeparatorSize;
				cnt += offset - naluSeparatorSize;
			}
			if (type == H264Utils::H264_NAL_TYPE_PIC_PARAM)
			{
				pps = buffer + (cnt - 5);
				size_t offset = 0;
				int naluSeparatorSize = 0;
				getNALUnitOffsetAndSizeBasedOnGivenType(buffer + (cnt - 1), length - (cnt - 1), offset, naluSeparatorSize, H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_NON_IDR_SLICE, false);
				if (!offset && !naluSeparatorSize)
				{
					ppsSize = length - (cnt - 5);
				}
				else
				{
					ppsSize = offset + 4 - naluSeparatorSize;
				}
				cnt += offset - naluSeparatorSize;
			}
		}
		cnt++;
	}

}