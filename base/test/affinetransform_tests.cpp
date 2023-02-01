#include <boost/test/unit_test.hpp>
#include "AffineTransform.h"
#include <opencv2/opencv.hpp>

BOOST_AUTO_TEST_SUITE(affinetransform_tests)

BOOST_AUTO_TEST_CASE(AffineTransformationTest)
{
    // Input test image
    cv::Mat testImage = cv::imread("test_image.jpg", cv::IMREAD_GRAYSCALE);

    // Create a 2x3 affine transformation matrix
    cv::Mat affineMatrix = (cv::Mat_<double>(2, 3) << 1, 0, 10, 0, 1, 20);

    // Apply affine transformation to the test image
    cv::Mat transformedImage;
    AffineTransform(testImage, transformedImage, affineMatrix);

    // Check if the size of the transformed image is as expected
    BOOST_CHECK_EQUAL(transformedImage.rows, testImage.rows);
    BOOST_CHECK_EQUAL(transformedImage.cols, testImage.cols);

    // Check if the pixel values in the transformed image are as expected
    for (int row = 0; row < transformedImage.rows; row++) {
        for (int col = 0; col < transformedImage.cols; col++) {
            BOOST_CHECK_EQUAL(transformedImage.at<uchar>(row, col), testImage.at<uchar>(row, col));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

