#include <boost/test/unit_test.hpp>

#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include <opencv2/cudawarping.hpp>

BOOST_AUTO_TEST_SUITE(opencvresize_tests, * boost::unit_test::disabled())

// opencv tests
void testopencvcpuresize()
{
    {
        auto img = cv::imread("./data/mono_1920x960.jpg", 0);
        if(!img.data)
        {
            std::cout << "image read failed" << std::endl;
            return;
        }
        std::cout << "hola" << std::endl;
        std::cout << img.rows << "<>" << img.cols << std::endl;
        std::cout << "hola" << std::endl;
        return;
    }

    std::string imgPath = "./data/mono_1920x960.jpg";

    auto src1 = cv::imread(imgPath, 0);
    if (!src1.data)
    {
        std::cout << "couldnot load image" << std::endl;
        return;
    }
    // cv::imshow("img", src1);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    cv::Mat dstMat2(src1.rows * 0.5, src1.cols * 0.5, CV_8UC1);
    cv::resize(src1, dstMat2, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC);

    cv::imwrite("./data/testOutput/testopencvcpuresize.jpg", dstMat2);
}

void testopencvgpuresize()
{
    std::string imgPath = "4k.jpg";

    auto src1 = cv::imread(imgPath, 0);
    if (!src1.data)
    {
        std::cout << "couldnot load image" << std::endl;
        return;
    }

    cv::Mat dst(src1.rows * 0.5, src1.cols * 0.5, CV_8UC1);

    cv::cuda::GpuMat dst_gpu(dst);
    cv::cuda::GpuMat src_gpu(src1);

    // auto cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    cv::cuda::resize(src_gpu, dst_gpu, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC);
    // cvStream.waitForCompletion();

    dst_gpu.download(dst);

    cv::imwrite("/home/al/Downloads/testopencvgpuresize.jpg", dst);
}

BOOST_AUTO_TEST_CASE(cpu)
{
    testopencvcpuresize();
}

BOOST_AUTO_TEST_CASE(gpu)
{
    testopencvgpuresize();
}

BOOST_AUTO_TEST_SUITE_END()