#ifndef __NV_JPEG_ENCODER_H__
#define __NV_JPEG_ENCODER_H__

#include <stdint.h>
#include<stdio.h>
#include "libjpeg-8b/jpeglib.h"

class JPEGEncoderL4TMHelper
{
public:
     JPEGEncoderL4TMHelper(int _quality);
     ~JPEGEncoderL4TMHelper();
	
     bool init(uint32_t width, uint32_t height, uint32_t stride, J_COLOR_SPACE color_space, double scale);
     int encode(const unsigned char* in_buf, unsigned char **out_buf, unsigned long &out_buf_size);

private:
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
 
    unsigned char **line[3];

 uint32_t comp_height[3];
    uint32_t comp_width[3];   
	 unsigned int stride[3];

 uint32_t h_max_samp = 0;
    uint32_t v_max_samp = 0;

 uint32_t h_samp[3];
    uint32_t v_samp[3];
 
 uint32_t planes; // this variable is used to distinguish between YUV420 and RGB

int quality;
   
};
#endif
