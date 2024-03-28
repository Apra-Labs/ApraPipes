#include "JPEGDecoderL4TMHelper.h"
#include <string.h>
#include <malloc.h>
#include <iostream>
#include <linux/videodev2.h>
#include "Logger.h"

JPEGDecoderL4TMHelper::JPEGDecoderL4TMHelper()
{

    memset(&cinfo, 0, sizeof(cinfo));
    memset(&jerr, 0, sizeof(jerr));
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);

    cinfo.do_fancy_upsampling = FALSE;
    cinfo.do_block_smoothing = FALSE;
    cinfo.dct_method = JDCT_FASTEST;
    // cinfo.bMeasure_ImageProcessTime = FALSE;
    cinfo.raw_data_out = TRUE;
}

JPEGDecoderL4TMHelper::~JPEGDecoderL4TMHelper()
{
    jpeg_destroy_decompress(&cinfo);
}

bool JPEGDecoderL4TMHelper::init(const unsigned char *in_buf, unsigned long in_buf_size, int &width, int &height)
{
    return true;

    // jpeg_mem_src(&cinfo, (unsigned char *)in_buf, in_buf_size);

    // (void)jpeg_read_header(&cinfo, TRUE);
    // cinfo.out_color_space = cinfo.jpeg_color_space;

    // if (cinfo.comp_info[0].h_samp_factor == 2)
    // {
    //     if (cinfo.comp_info[0].v_samp_factor == 2)
    //     {
    //         pixel_format = V4L2_PIX_FMT_YUV420M;
    //     }
    //     else
    //     {
    //         pixel_format = V4L2_PIX_FMT_YUV422M;
    //     }
    // }
    // else
    // {
    //     if (cinfo.comp_info[0].v_samp_factor == 1)
    //     {
    //         pixel_format = V4L2_PIX_FMT_YUV444M;
    //     }
    //     else
    //     {
    //         pixel_format = V4L2_PIX_FMT_YUV422M;
    //     }
    // }

    // jpeg_finish_decompress(&cinfo);
    // width = cinfo.image_width;
    // height = cinfo.image_height;

    // /* For some widths jpeglib requires more horizontal padding than I420
    //  * provides. In those cases we need to decode into separate buffers and then
    //  * copy over the data into our final picture buffer, otherwise jpeglib might
    //  * write over the end of a line into the beginning of the next line,
    //  * resulting in blocky artifacts on the left side of the picture. */
    // if (cinfo.output_width % (cinfo.max_h_samp_factor * DCTSIZE))
    // {
    //     // indirect method is not supported currently
    //     LOG_ERROR << "cinfo.output_width % (cinfo.max_h_samp_factor * DCTSIZE). output_width<" << cinfo.output_width << "> max_h_samp_factor<" << cinfo.max_h_samp_factor << "> DCTSIZE<" << DCTSIZE << ">";
    //     return false;
    // }

    // if (pixel_format != V4L2_PIX_FMT_YUV420M)
    // {
    //     LOG_ERROR << "piexel_format is expected to be V4L2_PIX_FMT_YUV420M. Actual<" << pixel_format << ">";
    //     return false;
    // }

    // return true;
}

int JPEGDecoderL4TMHelper::decode(const unsigned char *in_buf, unsigned long in_buf_size, unsigned char *out_buf)
{
    // Specify data source (in-memory buffer)
    jpeg_mem_src(&cinfo, const_cast<unsigned char*>(in_buf), in_buf_size);

    // Read JPEG header
    (void)jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = cinfo.jpeg_color_space;

    // Retrieve image dimensions
    // width = cinfo.image_width;
    // height = cinfo.image_height;

    // Start decompression
    jpeg_start_decompress(&cinfo);

    // Prepare buffer for a single scanline
    int row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    // Read scanlines and write decompressed data to the output buffer
    while (cinfo.output_scanline < cinfo.output_height)
    {
        // Read a scanline
        (void)jpeg_read_scanlines(&cinfo, buffer, 1);
        
        // Copy the scanline data to the output buffer
        memcpy(out_buf, buffer[0], row_stride);

        // Move to the next line in the output buffer
        out_buf += row_stride;
    }

    // Finish decompression
    jpeg_finish_decompress(&cinfo);

    return 0;
}
