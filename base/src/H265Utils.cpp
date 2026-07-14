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

bool H265Utils::getNALUnit(const char* buffer, size_t length, size_t& offset)
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

// typefound, vps(optional), sps(optional), pps(optional)
std::tuple<short, const_buffer, const_buffer, const_buffer> H265Utils::parseNalu(const const_buffer input)
{
	short typeFound = 0;
	char* p1 = reinterpret_cast<char*>(const_cast<void*>(input.data()));
	size_t offset = 0;
	typeFound = getNALUType(p1);

	if (isIDR((H265_NAL_TYPE)typeFound))
	{
		return { typeFound, const_buffer(), const_buffer(), const_buffer() };
	}

	if (typeFound == VPS)
	{
		size_t offset = 0;

		if (getNALUnit(p1, input.size(), offset)) // where does VPS start
		{
			p1 = p1 + offset;
			offset = 0;

			if (getNALUnit(p1, input.size(), offset)) // where does VPS end
			{
				char* vpsBits = p1;
				size_t nSize = offset - 3;
				if (p1[offset - 4] == 0x00)
					nSize--;
				size_t vpsSize = nSize;
				auto vpsBuffer = const_buffer(vpsBits, vpsSize);
				p1 = p1 + offset;

				if (getNALUnit(p1, input.size(), offset)) // SPS
				{
					char* spsBits = p1;
					size_t nSize = offset - 3;
					if (p1[offset - 4] == 0x00)
						nSize--;
					size_t spsSize = nSize;
					auto spsBuffer = const_buffer(spsBits, spsSize);
					p1 = p1 + offset;

					if (getNALUnit(p1, input.size(), offset)) // PPS
					{
						char* ppsBits = p1;
						size_t nSize = offset - 3;
						if (p1[offset - 4] == 0x00)
							nSize--;
						size_t ppsSize = nSize;
						auto ppsBuffer = const_buffer(ppsBits, ppsSize);

						typeFound = getNALUType(p1 + offset - 4);
						return { typeFound, vpsBuffer, spsBuffer, ppsBuffer };
					}
				}
			}
		}
	}

	// Handle SPS without VPS (some streams may not have VPS)
	if (typeFound == SPS)
	{
		size_t offset = 0;

		if (getNALUnit(p1, input.size(), offset)) // where does SPS start
		{
			p1 = p1 + offset;
			offset = 0;

			if (getNALUnit(p1, input.size(), offset)) // where does SPS end
			{
				char* spsBits = p1;
				size_t nSize = offset - 3;
				if (p1[offset - 4] == 0x00)
					nSize--;
				size_t spsSize = nSize;
				auto spsBuffer = const_buffer(spsBits, spsSize);
				p1 = p1 + offset;

				if (getNALUnit(p1, input.size(), offset)) // PPS
				{
					char* ppsBits = p1;
					size_t nSize = offset - 3;
					if (p1[offset - 4] == 0x00)
						nSize--;
					size_t ppsSize = nSize;
					auto ppsBuffer = const_buffer(ppsBits, ppsSize);

					typeFound = getNALUType(p1 + offset - 4);
					return { typeFound, const_buffer(), spsBuffer, ppsBuffer };
				}
			}
		}
	}

	return { typeFound, const_buffer(), const_buffer(), const_buffer() };
}