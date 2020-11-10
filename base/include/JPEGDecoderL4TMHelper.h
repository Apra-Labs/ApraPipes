#ifndef __NV_JPEG_ENCODER_H__
#define __NV_JPEG_ENCODER_H__

#include <stdint.h>
#include<stdio.h>
#include "libjpeg-8b/jpeglib.h"

class JPEGDecoderL4TMHelper
{
public:
     JPEGDecoderL4TMHelper();
     ~JPEGDecoderL4TMHelper();
	
     bool init(const unsigned char* in_buf, unsigned long in_buf_size, int& width, int& height);		
     int decode(const unsigned char* in_buf, unsigned long in_buf_size, unsigned char *out_buf);

private:

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

	uint32_t pixel_format = 0;

	unsigned char **line[3];
    unsigned char *y[4 * DCTSIZE] = { NULL, };
    unsigned char *u[4 * DCTSIZE] = { NULL, };
    unsigned char *v[4 * DCTSIZE] = { NULL, };
    int i, j;
    int lines, v_samp[3];
    unsigned char *base[3], *last[3];
    int stride[3];

};
#endif
