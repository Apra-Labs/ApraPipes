#include "JPEGEncoderL4TMHelper.h"
#include <string.h>
#include <malloc.h>

// https://dev.w3.org/Amaya/libjpeg/example.c

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ROUND_UP_4(num) (((num) + 3) & ~3)

JPEGEncoderL4TMHelper::JPEGEncoderL4TMHelper(int _quality): planes(3)
{
    memset(&cinfo, 0, sizeof(cinfo));
    memset(&jerr, 0, sizeof(jerr));
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_compress(&cinfo);
    jpeg_suppress_tables(&cinfo, TRUE);

    quality = _quality;
     for (auto i = 0; i < 3; i++)
    {
        line[i] = nullptr;
     }
}

JPEGEncoderL4TMHelper::~JPEGEncoderL4TMHelper()
{
    uint32_t i, j, k;
    for (i = 0; i < 3; i++)
    {
        if (line[i] != nullptr)
        {
            free(line[i]);
            line[i] = nullptr;
        }
    }
    jpeg_destroy_compress(&cinfo);
}

bool JPEGEncoderL4TMHelper::init(uint32_t width, uint32_t height, uint32_t _stride, J_COLOR_SPACE color_space, double scale)
{
    if (color_space == JCS_RGB)
    {
        planes = 1;
    }

    uint32_t i, j, k;

    comp_width[0] = width;
    comp_height[0] = height;
    stride[0] = _stride;

    comp_width[1] = width / 2;
    comp_height[1] = height / 2;
    stride[1] = _stride / 2;

    comp_width[2] = width / 2;
    comp_height[2] = height / 2;
    stride[2] = _stride / 2;

    h_max_samp = 0;
    v_max_samp = 0;

    for (i = 0; i < planes; ++i)
    {
        h_samp[i] = ROUND_UP_4(comp_width[0]) / comp_width[i];
        h_max_samp = MAX(h_max_samp, h_samp[i]);
        v_samp[i] = ROUND_UP_4(comp_height[0]) / comp_height[i];
        v_max_samp = MAX(v_max_samp, v_samp[i]);
    }

    for (i = 0; i < planes; ++i)
    {
        h_samp[i] = h_max_samp / h_samp[i];
        v_samp[i] = v_max_samp / v_samp[i];
    }

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3; // YUV RGB
    cinfo.in_color_space = color_space;

    if (scale != 1)
    {
        cinfo.image_scale = TRUE;
        cinfo.scaled_image_width = width * scale;
        cinfo.scaled_image_height = height * scale;
    }

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    if (planes == 3)
    {
        cinfo.raw_data_in = TRUE;
        for (i = 0; i < planes; i++)
        {
            cinfo.comp_info[i].h_samp_factor = h_samp[i];
            cinfo.comp_info[i].v_samp_factor = v_samp[i];
            line[i] = (unsigned char **)malloc(v_max_samp * DCTSIZE *
                                               sizeof(unsigned char *));
        }
    }   

    return true;
}

int JPEGEncoderL4TMHelper::encode(const unsigned char *in_buf, unsigned char **out_buf, unsigned long &out_buf_size)
{
    if(cinfo.in_color_space == JCS_RGB)
    {
        // every time destroy the cinfo and create it
        // if we don't do this per encode memory leak of the order of input frame
        jpeg_destroy_compress(&cinfo);

        memset(&cinfo, 0, sizeof(cinfo));
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_suppress_tables(&cinfo, TRUE);

        cinfo.image_width = comp_width[0];
        cinfo.image_height = comp_height[0];
        cinfo.input_components = 3; // YUV RGB
        cinfo.in_color_space = JCS_RGB;

        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality, TRUE);
    }

    jpeg_mem_dest(&cinfo, out_buf, &out_buf_size);
    jpeg_set_hardware_acceleration_parameters_enc(&cinfo, TRUE, out_buf_size, 0, 0);

    unsigned char *base[planes], *end[planes];

    uint32_t i, j, k;

    for (i = 0; i < planes; i++)
    {
        base[i] = (unsigned char *)in_buf;
        end[i] = base[i] + comp_height[i] * stride[i];

        in_buf = end[i];
    }

    jpeg_start_compress(&cinfo, TRUE);

    if (cinfo.err->msg_code)
    {
        char err_string[256];
        cinfo.err->format_message((j_common_ptr)&cinfo, err_string);
        // error message
        return -1;
    }

    if (planes == 3)
    {
        for (i = 0; i < comp_height[0]; i += v_max_samp * DCTSIZE)
        {
            for (k = 0; k < planes; k++)
            {
                for (j = 0; j < v_samp[k] * DCTSIZE; j++)
                {
                    line[k][j] = base[k];
                    if (base[k] + stride[k] < end[k])
                        base[k] += stride[k];
                }
            }

            jpeg_write_raw_data(&cinfo, line, v_max_samp * DCTSIZE);
        }
    }
    else
    {
        while (cinfo.next_scanline < cinfo.image_height)
        {
            auto row = base[0] + (cinfo.next_scanline*stride[0]);
            jpeg_write_scanlines(&cinfo, &row, 1); 
        }
    }
            

    jpeg_finish_compress(&cinfo);

    return 0;
}
