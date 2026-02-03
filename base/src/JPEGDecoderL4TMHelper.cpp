#include "JPEGDecoderL4TMHelper.h"
#include <string.h>
#include <malloc.h>
#include <iostream>
#include <linux/videodev2.h>
#include "Logger.h"
#include "nvbufsurface.h"
#include "jpegint.h"
#include <nvbufsurftransform.h>

JPEGDecoderL4TMHelper::JPEGDecoderL4TMHelper()
{

    memset(&cinfo, 0, sizeof(cinfo));
    memset(&jerr, 0, sizeof(jerr));
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);

    line[0] = y;
    line[1] = u;
    line[2] = v;

    pixel_format = V4L2_PIX_FMT_YUV420M;

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

    jpeg_mem_src(&cinfo, (unsigned char *)in_buf, in_buf_size);

    (void)jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = cinfo.jpeg_color_space;

    if (cinfo.comp_info[0].h_samp_factor == 2)
    {
        if (cinfo.comp_info[0].v_samp_factor == 2)
        {
            pixel_format = V4L2_PIX_FMT_YUV420M;
        }
        else
        {
            pixel_format = V4L2_PIX_FMT_YUV422M;
        }
    }
    else
    {
        if (cinfo.comp_info[0].v_samp_factor == 1)
        {
            pixel_format = V4L2_PIX_FMT_YUV444M;
        }
        else
        {
            pixel_format = V4L2_PIX_FMT_YUV422M;
        }
    }

   jpeg_finish_decompress(&cinfo);
    width = cinfo.image_width;
    height = cinfo.image_height;

    /* For some widths jpeglib requires more horizontal padding than I420
     * provides. In those cases we need to decode into separate buffers and then
     * copy over the data into our final picture buffer, otherwise jpeglib might
     * write over the end of a line into the beginning of the next line,
     * resulting in blocky artifacts on the left side of the picture. */
    if (cinfo.output_width % (cinfo.max_h_samp_factor * DCTSIZE))
    {
        // indirect method is not supported currently
        LOG_ERROR << "cinfo.output_width % (cinfo.max_h_samp_factor * DCTSIZE). output_width<" << cinfo.output_width << "> max_h_samp_factor<" << cinfo.max_h_samp_factor << "> DCTSIZE<" << DCTSIZE << ">";
        return false;
    }

    if (pixel_format != V4L2_PIX_FMT_YUV420M)
    {
        LOG_ERROR << "piexel_format is expected to be V4L2_PIX_FMT_YUV420M. Actual<" << pixel_format << ">";
        return false;
    }

    return true;
}



int JPEGDecoderL4TMHelper::decodeToFd(int &fd, unsigned char * in_buf,
        unsigned long in_buf_size)
{
    uint32_t pixel_format = 0;
    uint32_t buffer_id;
    NvBufSurface surface;

    if (in_buf == NULL || in_buf_size == 0)
    {
        LOG_INFO<<"Not decoding because input buffer = NULL or size = 0";
        return -1;
    }


    cinfo.out_color_space = JCS_YCbCr;

    jpeg_mem_src(&cinfo, in_buf, in_buf_size);

    cinfo.out_color_space = JCS_YCbCr;


    (void) jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_YCbCr;
    cinfo.IsVendorbuf = TRUE;
    cinfo.pVendor_buf = (unsigned char*)&surface;

    if (cinfo.comp_info[0].h_samp_factor == 2)
    {
        if (cinfo.comp_info[0].v_samp_factor == 2)
        {
            pixel_format = V4L2_PIX_FMT_YUV420M;
        }
        else
        {
            pixel_format = V4L2_PIX_FMT_YUV422M;
        }
    }
    else
    {
        if (cinfo.comp_info[0].v_samp_factor == 1)
        {
            pixel_format = V4L2_PIX_FMT_YUV444M;
        }
        else
        {
            pixel_format = V4L2_PIX_FMT_YUV422M;
        }
    }

    jpeg_start_decompress (&cinfo);

    if (cinfo.global_state != DSTATE_READY) {
        LOG_INFO<<"JPEG format is not supported by libnvjpeg";
        return -1;
    }

    jpeg_read_raw_data (&cinfo, NULL, cinfo.comp_info[0].v_samp_factor * DCTSIZE);
    jpeg_finish_decompress(&cinfo);

    // Image width and height should be even
    uint32_t width = (cinfo.image_width % 2 == 1) ? cinfo.image_width + 1 : cinfo.image_width;
    uint32_t height = (cinfo.image_height % 2 == 1) ? cinfo.image_height + 1 : cinfo.image_height;
    uint32_t pixfmt = pixel_format;
    int fd1= cinfo.fd;

    LOG_INFO<<"Succesfully decoded Buffer fd=" << fd;
     LOG_INFO<<"Succesfully decoded Buffer fd1=" << fd1;
     

    NvBufSurfTransformRect src_rect, dst_rect;
    src_rect.top = 0;
    src_rect.left = 0;
    src_rect.width = width;
    src_rect.height = height;
    dst_rect.top = 0;
    dst_rect.left = 0;
    dst_rect.width = width;
    dst_rect.height = height;

    // Set up transform parameters
    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(transform_params));
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform_params.transform_flip = NvBufSurfTransform_None;
    transform_params.transform_filter = NvBufSurfTransformInter_Nearest;
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;

    // auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());
    NvBufSurface *in_surf = nullptr;
    NvBufSurface *out_surf = nullptr;

    if (NvBufSurfaceFromFd(cinfo.fd, (void**)&in_surf) != 0) {
        LOG_INFO << "Failed to create input surface";
        return false;
        }

        if (NvBufSurfaceFromFd(fd, (void**)&out_surf) != 0) {
            LOG_INFO << "Failed to create output surface";
        return false;
        }

    NvBufSurfTransform_Error err = NvBufSurfTransform(in_surf, out_surf, &transform_params);

    if (err != NvBufSurfTransformError_Success) {
        LOG_INFO << "Transform failed============================================>>>>>>>>>>>>>>>>>>>>>>>>>>>error:" <<err;
    }



    return 0;
}

int JPEGDecoderL4TMHelper::decode(const unsigned char *in_buf, unsigned long in_buf_size, unsigned char *out_buf)
{
    jpeg_mem_src(&cinfo, (unsigned char *)in_buf, in_buf_size);
    (void)jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = cinfo.jpeg_color_space;

    for (i = 0; i < 3; i++)
    {
        auto tempheight = cinfo.image_height;
        v_samp[i] = cinfo.comp_info[i].v_samp_factor;
        if (i == 0)
        {
            stride[i] = cinfo.image_width;
            base[i] = out_buf;
        }
        else
        {
            tempheight = tempheight / 2;
            stride[i] = cinfo.image_width / 2;
            if (i == 1)
            {
                base[i] = out_buf + cinfo.image_width * cinfo.image_height;
            }
            else
            {
                base[i] = out_buf + (cinfo.image_width * cinfo.image_height) + (cinfo.image_width * cinfo.image_height) / 4;
            }
        }

        last[i] = base[i] + (stride[i] * (tempheight - 1));
    }

    jpeg_start_decompress(&cinfo);

    for (i = 0; i < (int)cinfo.image_height; i += v_samp[0] * DCTSIZE)
    {
        for (j = 0; j < (v_samp[0] * DCTSIZE); ++j)
        {
            /* Y */
            line[0][j] = base[0] + (i + j) * stride[0];

            /* U,V */
            if (pixel_format == V4L2_PIX_FMT_YUV420M)
            {
                /* Y */
                line[0][j] = base[0] + (i + j) * stride[0];
                if ((line[0][j] > last[0]))
                    line[0][j] = last[0];
                /* U */
                if (v_samp[1] == v_samp[0])
                {
                    line[1][j] = base[1] + ((i + j) / 2) * stride[1];
                }
                else if (j < (v_samp[1] * DCTSIZE))
                {
                    line[1][j] = base[1] + ((i / 2) + j) * stride[1];
                }
                if ((line[1][j] > last[1]))
                    line[1][j] = last[1];
                /* V */
                if (v_samp[2] == v_samp[0])
                {
                    line[2][j] = base[2] + ((i + j) / 2) * stride[2];
                }
                else if (j < (v_samp[2] * DCTSIZE))
                {
                    line[2][j] = base[2] + ((i / 2) + j) * stride[2];
                }
                if ((line[2][j] > last[2]))
                    line[2][j] = last[2];
            }
            else
            {
                line[1][j] = base[1] + (i + j) * stride[1];
                line[2][j] = base[2] + (i + j) * stride[2];
            }
        }

        lines = jpeg_read_raw_data(&cinfo, line, v_samp[0] * DCTSIZE);
        if ((!lines))
        {
            LOG_ERROR << "jpeg_read_raw_data() returned 0";
        }
    }

    jpeg_finish_decompress(&cinfo);

    return 0;
}
