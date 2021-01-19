#pragma once

#include <stddef.h>

class Frame;
class H264Utils {
private:
	H264Utils(void) {}
public:
		enum H264_NAL_TYPE {
			H264_NAL_TYPE_NON_IDR_SLICE = 1,
			H264_NAL_TYPE_DP_A_SLICE,
			H264_NAL_TYPE_DP_B_SLICE,
			H264_NAL_TYPE_DP_C_SLICE,
			H264_NAL_TYPE_IDR_SLICE,
			H264_NAL_TYPE_SEI,
			H264_NAL_TYPE_SEQ_PARAM,
			H264_NAL_TYPE_PIC_PARAM,
			H264_NAL_TYPE_ACCESS_UNIT,
			H264_NAL_TYPE_END_OF_SEQ,
			H264_NAL_TYPE_END_OF_STREAM,
			H264_NAL_TYPE_FILLER_DATA,
			H264_NAL_TYPE_SEQ_EXTENSION
		};
		static H264_NAL_TYPE getNALUType(const char *buffer);
		static H264_NAL_TYPE getNALUType(Frame *frm);
		static bool getNALUnit(const char *buffer, size_t length, size_t &offset);
};