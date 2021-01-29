#include <boost/test/unit_test.hpp>

#include "H264EncoderV4L2Helper.h"

#include "nvbuf_utils.h"

#include "test_utils.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(h264encoderv4l2helper_tests)

BOOST_AUTO_TEST_CASE(yuv420_black)
{
    auto width = 1280;
    auto height = 720;
    auto imageSizeY = width * height;
    auto imageSize = (imageSizeY * 3) >> 1;

    auto data = new uint8_t[imageSize];    
    memset(data, 0, imageSizeY);
    memset(data + imageSizeY, 128, imageSizeY >> 1);

    auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

    auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_YUV420M, width, height, width, 4*1024*1024, 30, [](frame_sp& frame) -> void {
        LOG_INFO << frame->size();
    } );

    for (auto i = 0; i < 100; i++)
    {
        helper->process(inputFrame);
    }

    boost::this_thread::sleep_for(boost::chrono::seconds(5));
    helper->stop();
    helper.reset();

    delete[] data;
}

BOOST_AUTO_TEST_CASE(yuv420_black_dmabuf)
{
    auto width = 1280;
    auto height = 720;
    auto imageSizeY = width * height;
    auto imageSize = (imageSizeY * 3) >> 1;

    auto data = new uint8_t[imageSize];    
    memset(data, 0, imageSizeY);
    memset(data + imageSizeY, 128, imageSizeY >> 1);


    NvBufferCreateParams inputParams = {0};

    inputParams.width = width;
    inputParams.height = height;
    inputParams.layout = NvBufferLayout_BlockLinear;
    inputParams.colorFormat = NvBufferColorFormat_NV12;
    inputParams.payloadType = NvBufferPayload_SurfArray;
    inputParams.nvbuf_tag = NvBufferTag_CAMERA;

    int fd = -1;
    BOOST_TEST(NvBufferCreateEx(&fd, &inputParams) == 0);
    BOOST_TEST(fd != -1);

    NvBufferParams par;
    NvBufferGetParams(fd, &par);
    void *ptr_y;
    uint8_t *ptr_cur;
    NvBufferMemMap(fd, 0, NvBufferMem_Write, &ptr_y);
    NvBufferMemSyncForCpu(fd, 0, &ptr_y);
    
    NvBufferMemSyncForDevice(fd, 0, &ptr_y);
    NvBufferMemUnMap(fd, 0, &ptr_y);

    auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(&fd, 4));

    auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_DMABUF, V4L2_PIX_FMT_YUV420M, width, height, width, 4*1024*1024, 30, [](frame_sp& frame) -> void {
        LOG_ERROR << frame->size();
    } );

    for (auto i = 0; i < 100; i++)
    {
        helper->process(inputFrame);
    }

    boost::this_thread::sleep_for(boost::chrono::seconds(5));
    helper->stop();
    helper.reset();

    BOOST_TEST(NvBufferDestroy(fd) == 0);
    delete[] data;
}

BOOST_AUTO_TEST_CASE(rgb24_black)
{
    auto width = 1280;
    auto height = 720;
    auto step = width * 3;
    auto imageSize = step * height;

    uint8_t* data;
    cudaMalloc(&data, imageSize);
    cudaMemset(data, 0, imageSize);
    cudaDeviceSynchronize();

    auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

    auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_RGB24, width, height, step, 4*1024*1024, 30, [](frame_sp& frame) -> void {
        LOG_INFO << frame->size();
    } );

    for (auto i = 0; i < 100; i++)
    {
        helper->process(inputFrame);
    }

    boost::this_thread::sleep_for(boost::chrono::seconds(5));
    helper->stop();
    helper.reset();

    cudaFree(data);
}

BOOST_AUTO_TEST_CASE(memory_cache_free_test)
{
    frame_sp cacheFrame;

    {
        auto width = 1280;
        auto height = 720;
        auto imageSizeY = width * height;
        auto imageSize = (imageSizeY * 3) >> 1;

        auto data = new uint8_t[imageSize];
        memset(data, 0, imageSizeY);
        memset(data + imageSizeY, 128, imageSizeY >> 1);

        auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

        auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_YUV420M, width, height, width, 4 * 1024 * 1024, 30, [&](frame_sp &frame) -> void {
            cacheFrame = frame;
        });

        for (auto i = 0; i < 10; i++)
        {
            helper->process(inputFrame);
        }

        boost::this_thread::sleep_for(boost::chrono::seconds(1));
        helper->stop();
        helper.reset();

        delete[] data;
    }

    LOG_ERROR << cacheFrame->data() << "<>" << cacheFrame->size();
    cacheFrame.reset();
}

BOOST_AUTO_TEST_SUITE_END()
