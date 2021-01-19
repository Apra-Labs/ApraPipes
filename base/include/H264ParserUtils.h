#pragma once

#include <stddef.h>

using namespace std;
struct sps_pps_properties {
	sps_pps_properties() {
		width = 640;
		height = 360;
		bitrate = 400000;
		fps = 30;
	}
	int width;
	int height;
	int bitrate;
	int fps;
};

// refeferd from https://stackoverflow.com/questions/12018535/get-the-width-height-of-the-video-from-h-264-nalu
class SpsPpsParsser{
public:
	SpsPpsParsser() {}
	void ParseSps(const unsigned char * pStart, unsigned short nLen, sps_pps_properties *output);
	void ParsePps(const unsigned char * pStart, unsigned short nLen, sps_pps_properties *output);
private:
	const unsigned char * m_pStart;
	unsigned short m_nLength;
	int m_nCurrentBit;
private:
	unsigned int ReadBit();
	unsigned int ReadBits(int n);
	unsigned int ReadExponentialGolombCode();
	unsigned int ReadSE();
};

class H264ParserUtils {
private:
	H264ParserUtils(void) {}
public:
	static void parse_sps(const char *sps, size_t len, sps_pps_properties *output);
	static void parse_pps(const char *pps, size_t len, sps_pps_properties *output);
};