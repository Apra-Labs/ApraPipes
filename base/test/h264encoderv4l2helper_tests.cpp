// #include <boost/test/unit_test.hpp>

// #include "RawImagePlanarMetadata.h"
// #include "H264EncoderV4L2Helper.h"
// #include "FrameFactory.h"
// #include "Frame.h"
// #include "Logger.h"

// #include "test_utils.h"

// #include "nvbuf_utils.h"

// BOOST_AUTO_TEST_SUITE(h264encoderv4l2helper_tests)

// BOOST_AUTO_TEST_CASE(yuv420_black)
// {
//     auto width = 1280;
//     auto height = 720;
//     auto imageSizeY = width * height;
//     auto imageSize = (imageSizeY * 3) >> 1;
//     std::string tempPinId = "temp"

//         auto data = new uint8_t[imageSize];
//     memset(data, 0, imageSizeY);
//     memset(data + imageSizeY, 128, imageSizeY >> 1);

//     auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

//     auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_YUV420M, width, height, width, 4 * 1024 * 1024, false, 0, 30, [](frame_sp &frame) -> void
//                                                 { LOG_INFO << frame->size(); });

//     for (auto i = 0; i < 100; i++)
//     {
//         helper->process(inputFrame);
//     }

//     boost::this_thread::sleep_for(boost::chrono::seconds(5));
//     helper->stop();
//     helper.reset();

//     delete[] data;
// }

// BOOST_AUTO_TEST_CASE(yuv420_black_dmabuf)
// {
//     auto width = 1280;
//     auto height = 720;

//     auto imageSize = (width * height * 3) >> 1;

//     auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
//     auto frameFactory = framefactory_sp(new FrameFactory(metadata, 10));

//     auto inputFrame = frameFactory->create(imageSize, frameFactory);

//     auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_DMABUF, V4L2_PIX_FMT_YUV420M, width, height, width, 4 * 1024 * 1024, 30, [](frame_sp &frame) -> void
//                                                 { LOG_DEBUG << frame->size(); });

//     for (auto i = 0; i < 100; i++)
//     {
//         helper->process(inputFrame);
//     }

//     boost::this_thread::sleep_for(boost::chrono::seconds(5));
//     helper->stop();
//     helper.reset();
// }

// BOOST_AUTO_TEST_CASE(rgb24_black)
// {
//     auto width = 1280;
//     auto height = 720;
//     auto step = width * 3;
//     auto imageSize = step * height;

//     uint8_t *data;
//     cudaMalloc(&data, imageSize);
//     cudaMemset(data, 0, imageSize);
//     cudaDeviceSynchronize();

//     auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

//     auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_RGB24, width, height, step, 4 * 1024 * 1024, 30, [](frame_sp &frame) -> void
//                                                 { LOG_INFO << frame->size(); });

//     for (auto i = 0; i < 100; i++)
//     {
//         helper->process(inputFrame);
//     }

//     boost::this_thread::sleep_for(boost::chrono::seconds(5));
//     helper->stop();
//     helper.reset();

//     cudaFree(data);
// }

// BOOST_AUTO_TEST_CASE(memory_cache_free_test)
// {
//     frame_sp cacheFrame;

//     {
//         auto width = 1280;
//         auto height = 720;
//         auto imageSizeY = width * height;
//         auto imageSize = (imageSizeY * 3) >> 1;

//         auto data = new uint8_t[imageSize];
//         memset(data, 0, imageSizeY);
//         memset(data + imageSizeY, 128, imageSizeY >> 1);

//         auto inputFrame = boost::shared_ptr<Frame>(new ExtFrame(data, imageSize));

//         auto helper = H264EncoderV4L2Helper::create(V4L2_MEMORY_MMAP, V4L2_PIX_FMT_YUV420M, width, height, width, 4 * 1024 * 1024, 30, [&](frame_sp &frame) -> void
//                                                     { cacheFrame = frame; });

//         for (auto i = 0; i < 10; i++)
//         {
//             helper->process(inputFrame);
//         }

//         boost::this_thread::sleep_for(boost::chrono::seconds(1));
//         helper->stop();
//         helper.reset();

//         delete[] data;
//     }

//     LOG_DEBUG << cacheFrame->data() << "<>" << cacheFrame->size();
//     cacheFrame.reset();
// }

// BOOST_AUTO_TEST_SUITE_END()
