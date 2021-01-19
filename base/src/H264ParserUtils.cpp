#include "stdafx.h"
#include "H264ParserUtils.h"
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#include <stdio.h>
#include <stdlib.h>    
#include <string.h>
#include <assert.h>
#include <iostream>

#include "Logger.h"

using namespace std;

// class SpsPpsParsser

unsigned int SpsPpsParsser::ReadBit()
{
	assert(m_nCurrentBit <= m_nLength * 8);
	int nIndex = m_nCurrentBit / 8;
	int nOffset = m_nCurrentBit % 8 + 1;

	m_nCurrentBit++;
	return (m_pStart[nIndex] >> (8 - nOffset)) & 0x01;
}

unsigned int SpsPpsParsser::ReadBits(int n)
{
	int r = 0;
	int i;
	for (i = 0; i < n; i++)
	{
		r |= (ReadBit() << (n - i - 1));
	}
	return r;
}

unsigned int SpsPpsParsser::ReadExponentialGolombCode()
{
	int r = 0;
	int i = 0;

	while ((ReadBit() == 0) && (i < 32))
	{
		i++;
	}

	r = ReadBits(i);
	r += (1 << i) - 1;
	return r;
}

unsigned int SpsPpsParsser::ReadSE()
{
	int r = ReadExponentialGolombCode();
	if (r & 0x01)
	{
		r = (r + 1) / 2;
	}
	else
	{
		r = -(r / 2);
	}
	return r;
}

void SpsPpsParsser::ParseSps(const unsigned char * pStart, unsigned short nLen, sps_pps_properties *output)
{
	LOG_TRACE << "ParseSps Len:" << nLen;
	m_pStart = pStart;
	m_nLength = nLen;
	m_nCurrentBit = 0;

	int frame_crop_left_offset = 0;
	int frame_crop_right_offset = 0;
	int frame_crop_top_offset = 0;
	int frame_crop_bottom_offset = 0;

	int profile_idc = ReadBits(8);
	int constraint_set0_flag = ReadBit();
	int constraint_set1_flag = ReadBit();
	int constraint_set2_flag = ReadBit();
	int constraint_set3_flag = ReadBit();
	int constraint_set4_flag = ReadBit();
	int constraint_set5_flag = ReadBit();
	int reserved_zero_2bits = ReadBits(2);
	int level_idc = ReadBits(8);
	int seq_parameter_set_id = ReadExponentialGolombCode();


	if (profile_idc == 100 || profile_idc == 110 ||
		profile_idc == 122 || profile_idc == 244 ||
		profile_idc == 44 || profile_idc == 83 ||
		profile_idc == 86 || profile_idc == 118)
	{
		int chroma_format_idc = ReadExponentialGolombCode();

		if (chroma_format_idc == 3)
		{
			int residual_colour_transform_flag = ReadBit();
		}
		int bit_depth_luma_minus8 = ReadExponentialGolombCode();
		int bit_depth_chroma_minus8 = ReadExponentialGolombCode();
		int qpprime_y_zero_transform_bypass_flag = ReadBit();
		int seq_scaling_matrix_present_flag = ReadBit();

		if (seq_scaling_matrix_present_flag)
		{
			int i = 0;
			for (i = 0; i < 8; i++)
			{
				int seq_scaling_list_present_flag = ReadBit();
				if (seq_scaling_list_present_flag)
				{
					int sizeOfScalingList = (i < 6) ? 16 : 64;
					int lastScale = 8;
					int nextScale = 8;
					int j = 0;
					for (j = 0; j < sizeOfScalingList; j++)
					{
						if (nextScale != 0)
						{
							int delta_scale = ReadSE();
							nextScale = (lastScale + delta_scale + 256) % 256;
						}
						lastScale = (nextScale == 0) ? lastScale : nextScale;
					}
				}
			}
		}
	}

	int log2_max_frame_num_minus4 = ReadExponentialGolombCode();
	int pic_order_cnt_type = ReadExponentialGolombCode();
	if (pic_order_cnt_type == 0)
	{
		int log2_max_pic_order_cnt_lsb_minus4 = ReadExponentialGolombCode();
	}
	else if (pic_order_cnt_type == 1)
	{
		int delta_pic_order_always_zero_flag = ReadBit();
		int offset_for_non_ref_pic = ReadSE();
		int offset_for_top_to_bottom_field = ReadSE();
		int num_ref_frames_in_pic_order_cnt_cycle = ReadExponentialGolombCode();
		int i;
		for (i = 0; i < num_ref_frames_in_pic_order_cnt_cycle; i++)
		{
			ReadSE();
			//sps->offset_for_ref_frame[ i ] = ReadSE();
		}
	}
	int max_num_ref_frames = ReadExponentialGolombCode();
	int gaps_in_frame_num_value_allowed_flag = ReadBit();
	int pic_width_in_mbs_minus1 = ReadExponentialGolombCode();
	int pic_height_in_map_units_minus1 = ReadExponentialGolombCode();
	int frame_mbs_only_flag = ReadBit();
	if (!frame_mbs_only_flag)
	{
		int mb_adaptive_frame_field_flag = ReadBit();
	}
	int direct_8x8_inference_flag = ReadBit();
	int frame_cropping_flag = ReadBit();
	if (frame_cropping_flag)
	{
		frame_crop_left_offset = ReadExponentialGolombCode();
		frame_crop_right_offset = ReadExponentialGolombCode();
		frame_crop_top_offset = ReadExponentialGolombCode();
		frame_crop_bottom_offset = ReadExponentialGolombCode();
	}
	int vui_parameters_present_flag = ReadBit();
	pStart++;

	int Width = ((pic_width_in_mbs_minus1 + 1) * 16) - frame_crop_right_offset * 2 - frame_crop_left_offset * 2;
	int Height = ((2 - frame_mbs_only_flag)* (pic_height_in_map_units_minus1 + 1) * 16) - (frame_crop_bottom_offset * 2) - (frame_crop_top_offset * 2);

	output->width = Width;
	output->height = Height;
	LOG_TRACE << "ParseSps Done W:" << Width <<" H:"<< Height;
}

void SpsPpsParsser::ParsePps(const unsigned char * pStart, unsigned short nLen, sps_pps_properties *output) {

}

// class SpsPpsParsser

void H264ParserUtils::parse_sps(const char *sps, size_t len, sps_pps_properties *output) {
	try {
		SpsPpsParsser parser;
		parser.ParseSps((unsigned char*)sps, len, output);
	}
	catch (std::exception &ex) {
		LOG_ERROR << "exception occured in parsing sps buffer";
	}
}

void H264ParserUtils::parse_pps(const char *pps, size_t len, sps_pps_properties *output) {
	try {
		SpsPpsParsser parser;
		parser.ParsePps((unsigned char*)pps, len, output);
	}
	catch (std::exception &ex) {
		LOG_ERROR << "exception occured in parsing pps buffer";
	}
}