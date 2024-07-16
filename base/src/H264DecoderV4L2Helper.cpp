/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions, and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
/**
 * Execution command:
 * ./decode_sample elementary_h264file.264 output_raw_file.yuv
**/
#include "DMAFDWrapper.h" 
#include "nvbuf_utils.h"
#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <cstdlib>
#include <libv4l2.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
#include "Logger.h"
 
#include "H264DecoderV4L2Helper.h"
#include "v4l2_nv_extensions.h"
 
/**
 *
 * V4L2 H264 Video Decoder Sample
 *
 * The video decoder device node isls
 *     /dev/nvhost-nvdec
 *
 * In this sample:
 * ## Pixel Formats
 * OUTPUT PLANE       | CAPTURE PLANE
 * :----------------: | :----------------:
 * V4L2_PIX_FMT_H264  | V4L2_PIX_FMT_NV12M
 *
 * ## Memory Type
 *            | OUTPUT PLANE        | CAPTURE PLANE
 * :--------: | :----------:        | :-----------:
 * MEMORY     | V4L2_MEMORY_MMAP    | V4L2_MEMORY_DMABUF
 *
 * ## Supported Controls
 * - #V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT
 * - V4L2_CID_MIN_BUFFERS_FOR_CAPTURE (Get the minimum buffers to be allocated
 * on capture plane.
 * Read-only. Valid after #V4L2_EVENT_RESOLUTION_CHANGE)
 *
 * ## Supported Events
 * Event                         | Purpose
 * ----------------------------- | :----------------------------:
 * #V4L2_EVENT_RESOLUTION_CHANGE | Resolution of the stream has changed.
 *
 * ## Opening the Decoder
 * The decoder device node is opened through the v4l2_open IOCTL call.
 * After opening the device, the application calls VIDIOC_QUERYCAP to identify
 * the driver capabilities.
 *
 * ## Subscribing events and setting up the planes
 * The application subscribes to the V4L2_EVENT_RESOLUTION_CHANGE event,
 * to detect the change in the resolution and handle the plane buffers
 * accordingly.
 * It calls VIDIOC_S_FMT to setup the formats required on
 * OUTPUT PLANE and CAPTURE PLANE for the data
 * negotiation between the former and the driver.
 *
 * ## Setting Controls
 * The application gets/sets the properties of the decoder by setting
 * the controls, calling VIDIOC_S_EXT_CTRLS, VIDIOC_G_CTRL.
 *
 * ## Buffer Management
 * Buffers are requested on the OUTPUT PLANE by the application, calling
 * VIDIOC_REQBUFS. The actual buffers allocated by the decoder are then
 * queried and exported as FD for the DMA-mapped buffer while mapped
 * for Mmaped buffer.
 * Status STREAMON is called on both planes to signal the decoder for
 * processing.
 *
 * Application continuously queues the encoded stream in the allocated
 * OUTPUT PLANE buffer and dequeues the next empty buffer fed into the
 * decoder.
 * The decoder decodes the buffer and triggers the resolution change event
 * on the capture plane
 *
 * ## Handling Resolution Change Events
 * When the decoder generates a V4L2_EVENT_RESOLUTION_CHANGE event, the
 * application calls STREAMOFF on the capture plane to tell the decoder to
 * deallocate the current buffers by calling REQBUF with count zero, get
 * the new capture plane format calling VIDIOC_G_FMT, and then proceed with
 * setting up the buffers for the capture plane.
 *
 * The decoding thread blocks on the DQ buffer call, which returns either after
 * a successful decoded raw buffer or after a specific timeout.
 *
 * ## EOS Handling
 * For sending EOS and receiving EOS from the decoder, the application must
 * - Send EOS to the decoder by queueing on the output plane a buffer with
 * bytesused = 0 for the 0th plane (`v4l2_buffer.m.planes[0].bytesused = 0`).
 * - Dequeues buffers on the output plane until it gets a buffer with bytesused = 0
 * for the 0th plane (`v4l2_buffer.m.planes[0].bytesused == 0`)
 * - Dequeues buffers on the capture plane until it gets a buffer with bytesused = 0
 * for the 0th plane.
 * After the last buffer on the capture plane is dequeued, set STREAMOFF on both
 * planes and destroy the allocated buffers.
 *
 */
 
Buffer::Buffer(enum v4l2_buf_type buf_type, enum v4l2_memory memory_type,
        uint32_t index)
        :buf_type(buf_type),
         memory_type(memory_type),
         index(index)
{
    uint32_t i;
 
    memset(planes, 0, sizeof(planes));
 
    mapped = false;
    n_planes = 1;
    for (i = 0; i < n_planes; i++)
    {
        this->planes[i].fd = -1;
        this->planes[i].data = NULL;
        this->planes[i].bytesused = 0;
        this->planes[i].mem_offset = 0;
        this->planes[i].length = 0;
        this->planes[i].fmt.sizeimage = 0;
    }
}
 
Buffer::Buffer(enum v4l2_buf_type buf_type, enum v4l2_memory memory_type,
        uint32_t n_planes, BufferPlaneFormat * fmt, uint32_t index)
        :buf_type(buf_type),
         memory_type(memory_type),
         index(index),
         n_planes(n_planes)
{
    uint32_t i;
 
    mapped = false;
 
    memset(planes, 0, sizeof(planes));
    for (i = 0; i < n_planes; i++)
    {
        this->planes[i].fd = -1;
        this->planes[i].fmt = fmt[i];
    }
}
 
Buffer::~Buffer()
{
    if (mapped)
    {
        unmap();
    }
}
 
int
Buffer::map()
{
    uint32_t j;
 
    if (memory_type != V4L2_MEMORY_MMAP)
    {
        LOG_INFO << "Buffer " << index << "already mapped" << endl;
        return -1;
    }
 
    if (mapped)
    {
        LOG_INFO << "Buffer " << index << "already mapped" << endl;
        return 0;
    }
 
    for (j = 0; j < n_planes; j++)
    {
        if (planes[j].fd == -1)
        {
            return -1;
        }
 
        planes[j].data = (unsigned char *) mmap(NULL,
                                                planes[j].length,
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED,
                                                planes[j].fd,
                                                planes[j].mem_offset);
        if (planes[j].data == MAP_FAILED)
        {
            LOG_INFO << "Could not map buffer " << index << ", plane " << j << endl;
            return -1;
        }
 
    }
    mapped = true;
    return 0;
}
 
void
Buffer::unmap()
{
    if (memory_type != V4L2_MEMORY_MMAP || !mapped)
    {
        LOG_INFO << "Cannot Unmap Buffer " << index <<
                ". Only mapped MMAP buffer can be unmapped" << endl;
        return;
    }
 
    for (uint32_t j = 0; j < n_planes; j++)
    {
        if (planes[j].data)
        {
            munmap(planes[j].data, planes[j].length);
        }
        planes[j].data = NULL;
    }
    mapped = false;
}
 
int
Buffer::fill_buffer_plane_format(uint32_t *num_planes,
        Buffer::BufferPlaneFormat *planefmts,
        uint32_t width, uint32_t height, uint32_t raw_pixfmt)
{
    switch (raw_pixfmt)
    {
        case V4L2_PIX_FMT_YUV420M:
            *num_planes = 3;
 
            planefmts[0].width = width;
            planefmts[1].width = width / 2;
            planefmts[2].width = width / 2;
 
            planefmts[0].height = height;
            planefmts[1].height = height / 2;
            planefmts[2].height = height / 2;
 
            planefmts[0].bytesperpixel = 1;
            planefmts[1].bytesperpixel = 1;
            planefmts[2].bytesperpixel = 1;
            break;
        case V4L2_PIX_FMT_NV12M:
            *num_planes = 2;
 
            planefmts[0].width = width;
            planefmts[1].width = width / 2;
 
            planefmts[0].height = height;
            planefmts[1].height = height / 2;
 
            planefmts[0].bytesperpixel = 1;
            planefmts[1].bytesperpixel = 2;
            break;
        default:
            LOG_INFO << "Unsupported pixel format " << raw_pixfmt << endl;
            return -1;
    }
    return 0;
}
 
void h264DecoderV4L2Helper::read_input_chunk_frame_sp(void* inputFrameBuffer, size_t inputFrameSize, Buffer * buffer)
{
    memcpy(buffer->planes[0].data,inputFrameBuffer,inputFrameSize);
    buffer->planes[0].bytesused = static_cast<uint32_t>(inputFrameSize);
}

/**
 * This function writes the video frame from the HW buffer
 * exported as FD into the destination file.
 * Using the FD, HW buffer parameters are filled by calling
 * NvBufferGetParams. The parameters received from the buffer are
 * then used to read the planar stream from the HW buffer into the
 * output filestream.
 *
 * For reading from the HW buffer:
 * A void data-pointer is created which stores the memory-mapped
 * virtual addresses of the planes.
 * For each plane, NvBufferMemMap is called which gets the
 * memory-mapped virtual address of the plane with the access
 * pointed by the flag into the void data-pointer.
 * Before the mapped memory is accessed, a call to NvBufferMemSyncForCpu()
 * with the virtual address returned must be present before any access is made
 * by the CPU to the buffer.
 *
 * After reading the data, the memory-mapped virtual address of the
 * plane is unmapped.
 */
 int h264DecoderV4L2Helper::sendFrames(int dmaOutFrameFd, frame_sp outputFrame)
{
   if (dmaOutFrameFd <= 0)
    {
        return -1;
    }
    outputFrame->timestamp = framesTimestampEntry.front();
    framesTimestampEntry.pop();

    send(outputFrame);

    return 0;
}
 
 int h264DecoderV4L2Helper::set_capture_plane_format(context_t * ctx, uint32_t pixfmt,
    uint32_t width, uint32_t height)
{
    int ret_val;
    struct v4l2_format format;
    uint32_t num_bufferplanes;
    Buffer::BufferPlaneFormat planefmts[MAX_PLANES];
 
    if (pixfmt != V4L2_PIX_FMT_NV12M)
    {
        LOG_ERROR << "Only V4L2_PIX_FMT_NV12M is supported" << endl;
        ctx->in_error = 1;
        return -1;
    }
    ctx->out_pixfmt = pixfmt;
    mBuffer->fill_buffer_plane_format(&num_bufferplanes, planefmts, width,
            height, pixfmt);
 
    ctx->cp_num_planes = num_bufferplanes;
    for (uint32_t j = 0; j < num_bufferplanes; ++j)
    {
        ctx->cp_planefmts[j] = planefmts[j];
    }
    memset(&format, 0, sizeof (struct v4l2_format));
    format.type = ctx->cp_buf_type;
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.num_planes = num_bufferplanes;
 
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_S_FMT, &format);
    if (ret_val)
    {
        LOG_ERROR << "Error in VIDIOC_S_FMT" << endl;
        ctx->in_error = 1;
    }
    else
    {
        ctx->cp_num_planes = format.fmt.pix_mp.num_planes;
        for (uint32_t j = 0; j < ctx->cp_num_planes; j++)
        {
            ctx->cp_planefmts[j].stride = format.fmt.pix_mp.plane_fmt[j].bytesperline;
            ctx->cp_planefmts[j].sizeimage = format.fmt.pix_mp.plane_fmt[j].sizeimage;
        }
    }
 
    return ret_val;
}
 
 void h264DecoderV4L2Helper::query_set_capture(context_t * ctx)
{
    struct v4l2_format format;
    struct v4l2_crop crop;
    int ret_val;
    int32_t min_cap_buffers;
    NvBufferCreateParams input_params = {0};
    NvBufferCreateParams cap_params = {0};
 
    /* Get format on capture plane set by device.
    ** This may change after an resolution change event.
    */
    format.type = ctx->cp_buf_type;
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_G_FMT, &format);
    if (ret_val)
    {
        LOG_ERROR << "Could not get format from decoder capture plane" << endl;
        ctx->in_error = 1;
        return ;
    }
 
    // Query cropping size and position.
 
    crop.type = ctx->cp_buf_type;
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_G_CROP, &crop);
    if (ret_val)
    {
        LOG_ERROR << "Could not get crop from decoder capture plane" << endl;
        ctx->in_error = 1;
        return;
    }
 
    LOG_INFO << "Resolution: " << crop.c.width << "x" << crop.c.height << endl;
    ctx->display_height = crop.c.height;
    ctx->display_width = crop.c.width;
 
    if (ctx->dst_dma_fd != -1)
    {
        NvBufferDestroy(ctx->dst_dma_fd);
        ctx->dst_dma_fd = -1;
    }
 
    pthread_mutex_lock(&ctx->queue_lock);
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_STREAMOFF, &ctx->cp_buf_type);
    if (ret_val)
    {
        ctx->in_error = 1;
    }
    else
    {
        pthread_cond_broadcast(&ctx->queue_cond);
    }
    pthread_mutex_unlock(&ctx->queue_lock);
 
    for (uint32_t j = 0; j < ctx->cp_num_buffers ; ++j)
    {
        switch (ctx->cp_mem_type)
        {
            case V4L2_MEMORY_MMAP:
                ctx->cp_buffers[j]->unmap();
                break;
            case V4L2_MEMORY_DMABUF:
                break;
            default:
                return;
        }
    }
 
    /* Request buffers with count 0 and destroy all
    ** previously allocated buffers.
    */
 
    ret_val = req_buffers_on_capture_plane(ctx, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        ctx->cp_mem_type, 0);
    if (ret_val)
    {
        LOG_ERROR << "Error in requesting 0 capture plane buffers" << endl;
        ctx->in_error = 1;
        return ;
    }
 
    // Destroy previous DMA buffers.
 
    if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
    {
        for (uint32_t index = 0 ; index < ctx->cp_num_buffers ; ++index)
        {
            if (ctx->dmabuff_fd[index] != 0)
            {
                ret_val = NvBufferDestroy(ctx->dmabuff_fd[index]);
                if (ret_val)
                {
                    LOG_ERROR << "Failed to Destroy NvBuffer" << endl;
                    ctx->in_error = 1;
                }
            }
        }
    }
 
    // Set capture plane format to update vars.
 
    ret_val = set_capture_plane_format(ctx, format.fmt.pix_mp.pixelformat,
        format.fmt.pix_mp.width, format.fmt.pix_mp.height);
    if (ret_val)
    {
        LOG_ERROR << "Error in setting capture plane format" << endl;
        ctx->in_error = 1;
        return ;
    }
 
    /* Get control value for min buffers which have to
    ** be requested on capture plane.
    */
 
    struct v4l2_control ctl;
    ctl.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;
 
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_G_CTRL, &ctl);
    if (ret_val)
    {
        LOG_ERROR << "Error getting value of control " << ctl.id << endl;
        ctx->in_error = 1;
        return ;
    }
    else
    {
        min_cap_buffers = ctl.value;
    }
 
    if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
    {
        if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
        {
            LOG_INFO << "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)" << endl;
            cap_params.colorFormat = NvBufferColorFormat_NV12;
        }
        else
        {
            LOG_INFO << "Decoder colorspace ITU-R BT.601 with extended range luma (0-255)" << endl;
            cap_params.colorFormat = NvBufferColorFormat_NV12_ER;
        }
 
        // Request number of buffers more than minimum returned by ctrl.
 
        ctx->cp_num_buffers = min_cap_buffers + 5;
 
        /* Create DMA Buffers by defining the parameters for the HW Buffer.
        ** @payloadType defines the memory handle for the NvBuffer, here
        ** defined for the set of planes.
        ** @nvbuf_tag identifies the type of device or component
        ** requesting the operation.
        ** @layout defines memory layout for the surfaces, either Pitch/BLockLinear.
        */
 
        for (uint32_t index = 0; index < ctx->cp_num_buffers; index++)
        {
            cap_params.width = crop.c.width;
            cap_params.height = crop.c.height;
            cap_params.layout = NvBufferLayout_BlockLinear;
            cap_params.payloadType = NvBufferPayload_SurfArray;
            cap_params.nvbuf_tag = NvBufferTag_VIDEO_DEC;
            ret_val = NvBufferCreateEx(&ctx->dmabuff_fd[index], &cap_params);
            if (ret_val)
            {
                LOG_ERROR << "Failed to create buffers" << endl;
                ctx->in_error = 1;
                break;
            }
        }
 
        // Request buffers on capture plane.
 
        ret_val = req_buffers_on_capture_plane(ctx, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        ctx->cp_mem_type, ctx->cp_num_buffers);
        if (ret_val)
        {
            LOG_ERROR << "Error in requesting capture plane buffers" << endl;
            ctx->in_error = 1;
            return ;
        }
 
    }
 
    // Enqueue all empty buffers on capture plane.
 
    for (uint32_t i = 0; i < ctx->cp_num_buffers; ++i)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
 
        memset(&v4l2_buf, 0, sizeof (v4l2_buf));
        memset(planes, 0, sizeof (planes));
 
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = ctx->cp_buf_type;
        v4l2_buf.memory = ctx->cp_mem_type;
        v4l2_buf.length = ctx->cp_num_planes;
        if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
        {
            // For NV12 format.
            v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[i];
            v4l2_buf.m.planes[1].m.fd = ctx->dmabuff_fd[i];
        }
 
        ret_val = q_buffer(ctx, v4l2_buf, NULL, ctx->cp_buf_type, ctx->cp_mem_type,
            ctx->cp_num_planes);
 
        if (ret_val)
        {
            LOG_ERROR << "Qing failed on capture plane" << endl;
            ctx->in_error = 1;
            return ;
        }
    }
 
    // Set streaming status ON on capture plane.
 
    ret_val = v4l2_ioctl(ctx->fd,VIDIOC_STREAMON, &ctx->cp_buf_type);
    if (ret_val != 0)
    {
        LOG_ERROR << "Streaming error on capture plane" << endl;
        ctx->in_error = 1;
    }
    ctx->cp_streamon = 1;
 
    LOG_DEBUG << "Query and set capture successful" << endl;
 
    return;
}
 
void * h264DecoderV4L2Helper::capture_thread(void *arg) 
{
    h264DecoderV4L2Helper *m_nThread = (h264DecoderV4L2Helper*)arg;
    context_t* ctx = &m_nThread->ctx;
    struct v4l2_event event;
    int ret_val;
 
    /* Need to wait for the first Resolution change event, so that
    ** the decoder knows the stream resolution and can allocate
    ** appropriate buffers when REQBUFS is called.
    */
 
    do
    {
        // Dequeue the subscribed event.
        ret_val = m_nThread->dq_event(ctx, event, 5000);
        if (ret_val)
        {
            if (errno == EAGAIN)
            {
                LOG_ERROR << "Timeout waiting for first V4L2_EVENT_RESOLUTION_CHANGE" << endl;
            }
            else
            {
                LOG_ERROR << "Error in dequeueing decoder event" << endl;
            }
            ctx->in_error = 1;
            break;
        }
    }
    while ((event.type != V4L2_EVENT_RESOLUTION_CHANGE) && !ctx->in_error);
 
    /* Recieved first resolution change event
    ** Format and buffers are now set on capture.
    */

    if (!ctx->in_error)
    {
       m_nThread->query_set_capture(ctx);
    }
 
    /* Check for resolution event to again
    ** set format and buffers on capture plane.
    */
    while (!(ctx->in_error || ctx->got_eos) || ctx->in_error)
    {
        Buffer *decoded_buffer = new Buffer(ctx->cp_buf_type, ctx->cp_mem_type, 0);
         ret_val = m_nThread->dq_event(ctx, event, 0);
        if (ret_val == 0)
        {
            switch (event.type)
            {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    m_nThread->query_set_capture(ctx);
                    continue; 
            }
        }
        // Main Capture loop for DQ and Q.
 
        while (!ctx->in_error)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
 
            memset(&v4l2_buf, 0, sizeof (v4l2_buf));
            memset(planes, 0, sizeof (planes));
            v4l2_buf.m.planes = planes;
 
            // Dequeue the filled buffer.
 
            if(m_nThread->dq_buffer(ctx, v4l2_buf, &decoded_buffer, ctx->cp_buf_type,
                 ctx->cp_mem_type, 0))
            {
                if (errno == EAGAIN)
                {
                    usleep(1000);
                }
                else
                {
                    ctx->in_error = 1;
                    LOG_ERROR << "Error while DQing at capture plane" << endl;
                }
                break;
            }
            
            if (ctx->display_width != 0)
            {
                /* Transformation parameters are defined
                ** which are passed to the NvBufferTransform
                ** for required conversion.
                */
                NvBufferRect src_rect, dest_rect;
                src_rect.top = 0;
                src_rect.left = 0;
                src_rect.width = ctx->display_width;
                src_rect.height = ctx->display_height;
                dest_rect.top = 0;
                dest_rect.left = 0;
                dest_rect.width = ctx->display_width;
                dest_rect.height = ctx->display_height;
 
                NvBufferTransformParams transform_params;
                memset(&transform_params,0,sizeof (transform_params));
 
                /* @transform_flag defines the flags for enabling the
                ** valid transforms. All the valid parameters are
                **  present in the nvbuf_utils header.
                */
                transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
                transform_params.transform_flip = NvBufferTransform_None;
                transform_params.transform_filter = NvBufferTransform_Filter_Smart;
                transform_params.src_rect = src_rect;
                transform_params.dst_rect = dest_rect;
 
                // Written for NV12.
                if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
                {
                    decoded_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];
                }
 
                /* Blocklinear to Pitch transformation is required
                ** to dump the raw decoded buffer data.
                */

                auto outputFrame = m_nThread->makeFrame();

                auto dmaOutFrame = static_cast<DMAFDWrapper *>(outputFrame->data());
                int f_d = dmaOutFrame->getFd();
                ret_val = NvBufferTransform(decoded_buffer->planes[0].fd,f_d, &transform_params);
                if (ret_val == -1)
                {
                    ctx->in_error = 1;
                    LOG_ERROR << "Transform failed" << endl;
                    break;
                }
 
                //send raw decoded  frame.
                m_nThread->sendFrames(f_d, outputFrame);

                if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
                {
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                    v4l2_buf.m.planes[1].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                }
 
                // Queue the buffer.
                ret_val = m_nThread->q_buffer(ctx, v4l2_buf, NULL, ctx->cp_buf_type, ctx->cp_mem_type,
                    ctx->cp_num_planes);
                if (ret_val)
                {
                    LOG_ERROR << "Qing failed on capture plane" << endl;
                    ctx->in_error = 1;
                    break;
                }
 
            }
            else
            {
                if (ctx->cp_mem_type == V4L2_MEMORY_DMABUF)
                {
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                    v4l2_buf.m.planes[1].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                }
 
                ret_val = m_nThread->q_buffer(ctx, v4l2_buf, NULL, ctx->cp_buf_type, ctx->cp_mem_type,
                    ctx->cp_num_planes);
                if (ret_val)
                {
                    LOG_ERROR << "Qing failed on capture plane" << endl;
                    ctx->in_error = 1;
                    break;
                }
            }
        }
   
    }
    LOG_TRACE << "Exiting decoder capture loop thread" << endl;
 
    return NULL;
}
 
 bool h264DecoderV4L2Helper::decode_process(context_t& ctx, void* inputFrameBuffer, size_t inputFrameSize)
{
    bool allow_DQ = true;
    int ret_val;
 
    /* As all the output plane buffers are queued, a buffer
    ** is dequeued first before new data is read and queued back.
    */
 
    if(ctx.eos == true)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        Buffer *buffer;
 
        memset(&v4l2_buf, 0, sizeof (v4l2_buf));
        memset(planes, 0, sizeof (planes));
 
        v4l2_buf.m.planes = planes;
        if (allow_DQ)
        {
            // Dequeue the empty buffer on output plane.
            ret_val = dq_buffer(&ctx, v4l2_buf, &buffer, ctx.op_buf_type,
                ctx.op_mem_type, -1);
            if (ret_val)
            {
                LOG_ERROR << "Error DQing buffer at output plane" << endl;
                ctx.in_error = 1;
            }
        }
        else
        {
            allow_DQ = true;
            buffer = ctx.op_buffers[v4l2_buf.index];
        }
 
        // Read and enqueue the filled buffer.
 
        if (ctx.decode_pixfmt == V4L2_PIX_FMT_H264)
        {
            read_input_chunk_frame_sp(inputFrameBuffer, inputFrameSize, buffer);
        }
        else
        {
            LOG_INFO << "Currently only H264 supported" << endl;
            ctx.in_error = 1;
        }
 
        ret_val = q_buffer(&ctx, v4l2_buf, buffer,
            ctx.op_buf_type, ctx.op_mem_type, ctx.op_num_planes);
        if (ret_val)
        {
            LOG_ERROR << "Error Qing buffer at output plane" << endl;
            ctx.in_error = 1;
        }

        if((!ctx.eos && !ctx.in_error))
        {
            if (v4l2_buf.m.planes[0].bytesused == 0)
            {
            ctx.eos = true;
            LOG_INFO << "Input file read complete" << endl;
            }
        }
    }
    return ctx.eos;
}
 
 int h264DecoderV4L2Helper::dq_event(context_t * ctx, struct v4l2_event &event, uint32_t max_wait_ms)
{
    int ret_val;
    do
    {
        ret_val = v4l2_ioctl(ctx->fd, VIDIOC_DQEVENT, &event);
 
        if (errno != EAGAIN)
        {
            break;
        }
        else if (max_wait_ms-- == 0)
        {
            break;
        }
        else
        {
            usleep(1000);
        }
    }
    while (ret_val && (ctx->op_streamon || ctx->cp_streamon));
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::dq_buffer(context_t * ctx, struct v4l2_buffer &v4l2_buf, Buffer ** buffer,
    enum v4l2_buf_type buf_type, enum v4l2_memory memory_type, uint32_t num_retries)
{
    int ret_val;
    bool is_in_error = false;
    v4l2_buf.type = buf_type;
    v4l2_buf.memory = memory_type;
    do
    {
        ret_val = v4l2_ioctl(ctx->fd, VIDIOC_DQBUF, &v4l2_buf);

        if (ret_val == 0)
        {
            pthread_mutex_lock(&ctx->queue_lock);
            switch (v4l2_buf.type)
            {
                case V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE:
                    if (buffer)
                        *buffer = ctx->op_buffers[v4l2_buf.index];
                    for (uint32_t j = 0; j < ctx->op_buffers[v4l2_buf.index]->n_planes; j++)
                    {
                        ctx->op_buffers[v4l2_buf.index]->planes[j].bytesused =
                        v4l2_buf.m.planes[j].bytesused;
                    }
                    ctx->num_queued_op_buffers--;
                    break;
 
                case V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE:
                    if (buffer)
                        *buffer = ctx->cp_buffers[v4l2_buf.index];
                    for (uint32_t j = 0; j < ctx->cp_buffers[v4l2_buf.index]->n_planes; j++)
                    {
                        ctx->cp_buffers[v4l2_buf.index]->planes[j].bytesused =
                        v4l2_buf.m.planes[j].bytesused;
                    }
                    break;
 
                default:
                    LOG_INFO << "Invaild buffer type" << endl;
            }
            pthread_cond_broadcast(&ctx->queue_cond);
            pthread_mutex_unlock(&ctx->queue_lock);
        }
        else if (errno == EAGAIN)
        {
            pthread_mutex_lock(&ctx->queue_lock);
            if (v4l2_buf.flags & V4L2_BUF_FLAG_LAST)
            {
                pthread_mutex_unlock(&ctx->queue_lock);
                break;
            }
            pthread_mutex_unlock(&ctx->queue_lock);
 
            if (num_retries-- == 0)
            {
                // Resource temporarily unavailable.
                LOG_INFO << "Resource unavailable" << endl;
                break;
            }
        }
        else
        {
            is_in_error = 1;
            break;
        }
    }
    while (ret_val && !is_in_error);
    return ret_val;
}
 
 int h264DecoderV4L2Helper::q_buffer(context_t * ctx, struct v4l2_buffer &v4l2_buf, Buffer * buffer,
    enum v4l2_buf_type buf_type, enum v4l2_memory memory_type, int num_planes)
{
    int ret_val;
    uint32_t j;
 
    pthread_mutex_lock(&ctx->queue_lock);
    v4l2_buf.type = buf_type;
    v4l2_buf.memory = memory_type;
    v4l2_buf.length = num_planes;
 
    switch (memory_type)
    {
        case V4L2_MEMORY_MMAP:
            for (j = 0; j < buffer->n_planes; ++j)
            {
                v4l2_buf.m.planes[j].bytesused =
                buffer->planes[j].bytesused;
            }
            break;
 
        case V4L2_MEMORY_DMABUF:
            break;
 
        default:
            pthread_cond_broadcast(&ctx->queue_cond);
            pthread_mutex_unlock(&ctx->queue_lock);
            return -1;
    }
 
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_QBUF, &v4l2_buf);
 
    if (ret_val == 0)
    {
        if (v4l2_buf.type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
        {
            ctx->num_queued_op_buffers++;
        }
        pthread_cond_broadcast(&ctx->queue_cond);
    }
    pthread_mutex_unlock(&ctx->queue_lock);
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::req_buffers_on_capture_plane(context_t * ctx, enum v4l2_buf_type buf_type, enum v4l2_memory mem_type,
    int num_buffers)
{
    struct v4l2_requestbuffers reqbufs;
    int ret_val;
    memset(&reqbufs, 0, sizeof (struct v4l2_requestbuffers));
 
    reqbufs.count = num_buffers;
    reqbufs.memory = mem_type;
    reqbufs.type = buf_type;
 
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs);
    if (ret_val)
        return ret_val;
 
    if (reqbufs.count)
    {
        ctx->cp_buffers = new Buffer *[reqbufs.count];
        for (uint32_t i = 0; i < reqbufs.count; ++i)
        {
            ctx->cp_buffers[i] = new Buffer(buf_type, mem_type,
                ctx->cp_num_planes, ctx->cp_planefmts, i);
        }
    }
    else
    {
        for (uint32_t i = 0; i < ctx->cp_num_buffers; ++i)
        {
            delete ctx->cp_buffers[i];
        }
        delete[] ctx->cp_buffers;
        ctx->cp_buffers = NULL;
    }
    ctx->cp_num_buffers = reqbufs.count;
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::req_buffers_on_output_plane(context_t * ctx, enum v4l2_buf_type buf_type, enum v4l2_memory mem_type,
    int num_buffers)
{
    struct v4l2_requestbuffers reqbufs;
    int ret_val;
    memset(&reqbufs, 0, sizeof (struct v4l2_requestbuffers));
 
    reqbufs.count = num_buffers;
    reqbufs.memory = mem_type;
    reqbufs.type = buf_type;
 
    ret_val = v4l2_ioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs);
    if (ret_val)
        return ret_val;
 
    if (reqbufs.count)
    {
        ctx->op_buffers = new Buffer *[reqbufs.count];
        for (uint32_t i = 0; i < reqbufs.count; ++i)
        {
            ctx->op_buffers[i] = new Buffer(buf_type, mem_type,
                ctx->op_num_planes, ctx->op_planefmts, i);
        }
    }
    else
    { 
        for (uint32_t i = 0; i < ctx->op_num_buffers; ++i)
        {
            delete ctx->op_buffers[i];
        }
        delete[] ctx->op_buffers;
        ctx->op_buffers = NULL;
    }
    ctx->op_num_buffers = reqbufs.count;
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::set_output_plane_format(context_t& ctx, uint32_t pixfmt, uint32_t sizeimage)
{
    int ret_val;
    struct v4l2_format format;
 
    memset(&format, 0, sizeof (struct v4l2_format));
    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.num_planes = 1;
    format.fmt.pix_mp.plane_fmt[0].sizeimage = sizeimage;
 
    ret_val = v4l2_ioctl(ctx.fd, VIDIOC_S_FMT, &format);
 
    if (ret_val == 0)
    {
        ctx.op_num_planes = format.fmt.pix_mp.num_planes;
        for (uint32_t i = 0; i < ctx.op_num_planes; ++i)
        {
            ctx.op_planefmts[i].stride = format.fmt.pix_mp.plane_fmt[i].bytesperline;
            ctx.op_planefmts[i].sizeimage = format.fmt.pix_mp.plane_fmt[i].sizeimage;
        }
    }
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::set_ext_controls(int fd, uint32_t id, uint32_t value)
{
    int ret_val;
    struct v4l2_ext_control ctl;
    struct v4l2_ext_controls ctrls;
 
    memset(&ctl, 0, sizeof (struct v4l2_ext_control));
    memset(&ctrls, 0, sizeof (struct v4l2_ext_controls));
    ctl.id = id;
    ctl.value = value;
    ctrls.controls = &ctl;
    ctrls.count = 1;
 
    ret_val = v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
 
    return ret_val;
}
 
int h264DecoderV4L2Helper::subscribe_event(int fd, uint32_t type, uint32_t id, uint32_t flags)
{
    struct v4l2_event_subscription sub;
    int ret_val;
 
    memset(&sub, 0, sizeof (struct v4l2_event_subscription));
 
    sub.type = type;
    sub.id = id;
    sub.flags = flags;
 
    ret_val = v4l2_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
 
    return ret_val;
}
 
bool h264DecoderV4L2Helper::init(std::function<void(frame_sp&)> _send, std::function<frame_sp()> _makeFrame)
{
    makeFrame = _makeFrame;
    mBuffer.reset(new Buffer());
    send =  _send;
    return initializeDecoder();
}
bool h264DecoderV4L2Helper::initializeDecoder()
{
    int flags = 0;
    struct v4l2_capability caps;
    struct v4l2_buffer op_v4l2_buf;
    struct v4l2_plane op_planes[MAX_PLANES];
    struct v4l2_exportbuffer op_expbuf;
 
    memset(&ctx, 0, sizeof (context_t));
    ctx.out_pixfmt = V4L2_PIX_FMT_NV12;
    ctx.decode_pixfmt = V4L2_PIX_FMT_H264;
    ctx.op_mem_type = V4L2_MEMORY_MMAP;
    ctx.cp_mem_type = V4L2_MEMORY_DMABUF;
    ctx.op_buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    ctx.cp_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ctx.fd = -1;
    ctx.dst_dma_fd = -1;
    ctx.num_queued_op_buffers = 0;
    ctx.op_buffers = NULL;
    ctx.cp_buffers = NULL;
    pthread_mutex_init(&ctx.queue_lock, NULL);
    pthread_cond_init(&ctx.queue_cond, NULL);
 
    /* The call creates a new V4L2 Video Decoder object
    ** on the device node "/dev/nvhost-nvdec"
    ** Additional flags can also be given with which the device
    ** should be opened.
    ** This opens the device in Blocking mode.
    */
    ctx.fd = v4l2_open(DECODER_DEV, flags | O_RDWR);
    
    if (ctx.fd == -1)
    {
        LOG_ERROR << "Could not open device" << DECODER_DEV << endl;
        ctx.in_error = 1;
    }
 
    /* The Querycap Ioctl call queries the video capabilities
    ** of the opened node and checks for
    ** V4L2_CAP_VIDEO_M2M_MPLANE capability on the device.
    */
 
    ret = v4l2_ioctl(ctx.fd, VIDIOC_QUERYCAP, &caps);
    if (ret)
    {
        LOG_ERROR << "Failed to query video capabilities" << endl;
        ctx.in_error = 1;
    }
    if (!(caps.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE))
    {
        LOG_ERROR << "Device does not support V4L2_CAP_VIDEO_M2M_MPLANE" << endl;
        ctx.in_error = 1;
    }
 
    /* Subscribe to Resolution change event.
    ** This is required to catch whenever resolution change event
    ** is triggered to set the format on capture plane.
    */
 
    ret = subscribe_event(ctx.fd, V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    if (ret)
    {
        LOG_ERROR << "Failed to subscribe for resolution change" << endl;
        ctx.in_error = 1;
    }
 
    /* Set appropriate controls.
    ** V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control is
    ** set to false so that application can send chunks of encoded
    ** data instead of forming complete frames.
    */
 
    ret = set_ext_controls(ctx.fd, V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT, 1);
    if (ret)
    {
        LOG_ERROR << "Failed to set control disable complete frame" << endl;
        ctx.in_error = 1;
    }
 
    /* Set format on output plane.
    ** The format of the encoded bitstream is set.
    */
 
    ret = set_output_plane_format(ctx, ctx.decode_pixfmt, CHUNK_SIZE);
    if (ret)
    {
        LOG_ERROR << "Error in setting output plane format" << endl;
        ctx.in_error = 1;
    }
 

    // Should not be a part of init
    /* Request buffers on output plane to fill
    ** the input bitstream.
    */
 
    ret = req_buffers_on_output_plane(&ctx, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
        ctx.op_mem_type, 1); //maxBufferr should come from props
    if (ret)
    {
        LOG_ERROR << "Error in requesting buffers on output plane" << endl;
        ctx.in_error = 1;
    }
 
    /* Query the status of requested buffers.
    ** For each requested buffer, export buffer
    ** and map it for MMAP memory.
    */
 
    for (uint32_t i = 0; i < ctx.op_num_buffers; ++i)
    {
        memset(&op_v4l2_buf, 0, sizeof (struct v4l2_buffer));
        memset(op_planes, 0, sizeof (op_planes));
        op_v4l2_buf.index = i;
        op_v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        op_v4l2_buf.memory = ctx.op_mem_type;
        op_v4l2_buf.m.planes = op_planes;
        op_v4l2_buf.length = ctx.op_num_planes;

        ret = v4l2_ioctl(ctx.fd, VIDIOC_QUERYBUF, &op_v4l2_buf);
        if (ret)
        {
            LOG_ERROR << "Error in querying buffers" << endl;
            ctx.in_error = 1;
        }
 
        for (uint32_t j = 0; j < op_v4l2_buf.length; ++j)
        {
            ctx.op_buffers[i]->planes[j].length =
            op_v4l2_buf.m.planes[j].length;
            ctx.op_buffers[i]->planes[j].mem_offset =
            op_v4l2_buf.m.planes[j].m.mem_offset;
        }
 
        memset(&op_expbuf, 0, sizeof (struct v4l2_exportbuffer));
        op_expbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        op_expbuf.index = i;
 
        for (uint32_t j = 0; j < ctx.op_num_planes; ++j)
        {
            op_expbuf.plane = j;
            ret = v4l2_ioctl(ctx.fd, VIDIOC_EXPBUF, &op_expbuf);
            if (ret)
            {
                LOG_ERROR << "Error in exporting buffer at index" << i << endl;
                ctx.in_error = 1;
            }
            ctx.op_buffers[i]->planes[j].fd = op_expbuf.fd;
        }
 
        if (ctx.op_buffers[i]->map())
        {
            LOG_ERROR << "Buffer mapping error on output plane" << endl;
            ctx.in_error = 1;
        }
    }
 
    /* Start stream processing on output plane
    ** by setting the streaming status ON.
    */
 
    ret = v4l2_ioctl(ctx.fd,VIDIOC_STREAMON, &ctx.op_buf_type);
    if (ret != 0)
    {
        LOG_ERROR << "Streaming error on output plane" << endl;
        ctx.in_error = 1;
    }
 
    ctx.op_streamon = 1;
    // Create Capture loop thread.
    typedef void * (*THREADFUNCPTR)(void *);
   
    pthread_create(&ctx.dec_capture_thread, NULL,h264DecoderV4L2Helper::capture_thread, (void *) (this));

    return true;
}
int h264DecoderV4L2Helper::process(void* inputFrameBuffer, size_t inputFrameSize, uint64_t inputFrameTS)
{
    uint32_t idx = 0;
    if(inputFrameSize)
	framesTimestampEntry.push(inputFrameTS);

    if((inputFrameSize && ctx.eos && ctx.got_eos) || ctx.in_error)
    {
        ctx.in_error = false;
        deQueAllBuffers();
        ctx.eos = false;
        ctx.got_eos = false;
        initializeDecoder();
    }

    while (!ctx.eos && !ctx.in_error && idx < ctx.op_num_buffers)
    {
        struct v4l2_buffer queue_v4l2_buf_op;
        struct v4l2_plane queue_op_planes[MAX_PLANES];
        Buffer *buffer;
 
        memset(&queue_v4l2_buf_op, 0, sizeof (queue_v4l2_buf_op));
        memset(queue_op_planes, 0, sizeof (queue_op_planes));
 
        buffer = ctx.op_buffers[idx];
        if (ctx.decode_pixfmt == V4L2_PIX_FMT_H264)
        {
            read_input_chunk_frame_sp(inputFrameBuffer, inputFrameSize, buffer);
        }
        else
        {
            LOG_ERROR << "Currently only H264 supported" << endl;
            ctx.in_error = 1;
        }
 
        queue_v4l2_buf_op.index = idx;
        queue_v4l2_buf_op.m.planes = queue_op_planes;
 
        /* Enqueue the buffer on output plane
        ** It is necessary to queue an empty buffer
        ** to signal EOS to the decoder.
        */
        int qBuffer = 0;
        int counter = 0;
        do
        {
            counter++;
            qBuffer = q_buffer(&ctx, queue_v4l2_buf_op, buffer,
            ctx.op_buf_type, ctx.op_mem_type, ctx.op_num_planes);
            if(counter > 1)
            {
                LOG_INFO << "Unable to queue buffers " << qBuffer;
            }
        }
        while(qBuffer);
 
        if (queue_v4l2_buf_op.m.planes[0].bytesused == 0)
        {
            ctx.eos = true;
            LOG_DEBUG << "Input file read complete" << endl;
            break;
        }
        idx++;
    }
 
    // Dequeue and queue loop on output plane.
    ctx.eos = decode_process(ctx,inputFrameBuffer, inputFrameSize);
   
    /* For blocking mode, after getting EOS on output plane,
    ** dequeue all the queued buffers on output plane.
    ** After that capture plane loop should be signalled to stop.
    *///
 
    while (ctx.num_queued_op_buffers > 0 && !ctx.in_error && !ctx.got_eos)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
 
        memset(&v4l2_buf, 0, sizeof (v4l2_buf));
        memset(planes, 0, sizeof (planes));
 
        v4l2_buf.m.planes = planes;
        ret = dq_buffer(&ctx, v4l2_buf, NULL, ctx.op_buf_type, ctx.op_mem_type, -1);
        if (ret)
        {
            LOG_ERROR << "Error DQing buffer at output plane" << endl;
            ctx.in_error = 1;
            break;
            return true;
        }
    }
 
    // Signal EOS to capture loop thread to exit.
 
    if(ctx.eos == true)
    {
        ctx.got_eos = 1;
    }
    return true;
}
void h264DecoderV4L2Helper::closeAllThreads(frame_sp eosFrame) 
{
    process(eosFrame->data(), eosFrame->size(), 0);
    deQueAllBuffers();
}

void h264DecoderV4L2Helper::deQueAllBuffers()
{
    if (ctx.fd != -1)
    {
        if (ctx.dec_capture_thread)
        {
            pthread_join(ctx.dec_capture_thread, NULL);
        }
 
        // All the allocated DMA buffers must be destroyed.
        if (ctx.cp_mem_type == V4L2_MEMORY_DMABUF)
        {
            for (uint32_t idx = 0; idx < ctx.cp_num_buffers; ++idx)
            {
                ret = NvBufferDestroy(ctx.dmabuff_fd[idx]);
                if (ret)
                {
                    LOG_ERROR << "Failed to Destroy Buffers" << endl;
                }
            }
        }
 
        // Stream off on both planes.
 
        ret = v4l2_ioctl(ctx.fd, VIDIOC_STREAMOFF, &ctx.op_buf_type);
        ret = v4l2_ioctl(ctx.fd, VIDIOC_STREAMOFF, &ctx.cp_buf_type);
 
        // Unmap MMAPed buffers.
 
        for (uint32_t i = 0; i < ctx.op_num_buffers; ++i)
        {
            ctx.op_buffers[i]->unmap();
        }
 
        // Request 0 buffers on both planes.
 
        ret = req_buffers_on_output_plane(&ctx, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            ctx.op_mem_type, 0);
        ret = req_buffers_on_capture_plane(&ctx, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            ctx.cp_mem_type, 0);
 
        // Destroy DMA buffers.
 
        if (ctx.cp_mem_type == V4L2_MEMORY_DMABUF)
        {
            for (uint32_t i = 0; i < ctx.cp_num_buffers; ++i)
            {
                if (ctx.dmabuff_fd[i] != 0)
                {
                    ret = NvBufferDestroy(ctx.dmabuff_fd[i]);
                    ctx.dmabuff_fd[i] = 0;
                    if (ret < 0)
                    {
                        LOG_ERROR << "Failed to destroy buffer" << endl;
                    }
                }
            }
        }
        if (ctx.dst_dma_fd != -1)
        {
            NvBufferDestroy(ctx.dst_dma_fd);
            ctx.dst_dma_fd = -1;
        }
 
        // Close the opened V4L2 device.
 
        ret = v4l2_close(ctx.fd);
        if (ret)
        {
            LOG_ERROR << "Unable to close the device" << endl;
            ctx.in_error = 1;
        }
    }
 
    // Report application run status on exit.
    if (ctx.in_error)
    {
        LOG_ERROR << "Decoder Run failed" <<  endl;
    }
    else
    {
        LOG_DEBUG  << "Decoder Run is successful" << endl;
    }
 
    return;
}
 
 
 

