#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/pool/object_pool.hpp>
#include "Frame.h"
#include "Frame.h"
#include "CommonDefs.h"

BOOST_AUTO_TEST_SUITE(cv_memory_leaks_tests)


BOOST_AUTO_TEST_CASE(cv_mat_memory_leak)
{
	cv::Mat();
}

BOOST_AUTO_TEST_CASE(cv_mat_memory_leak_2, *boost::unit_test::disabled())
{
	// Disabled: cv::imshow requires GUI support not available in CI
	auto zeros = cv::Mat::zeros(cv::Size(1920, 454), CV_8UC1);

	for (auto i = 0; i < 100; i++)
	{
#ifndef LINUX
		cv::imshow("", zeros);
		cv::waitKey(1);
#endif
	}
}


BOOST_AUTO_TEST_CASE(cv_mat_memory_leak_3)
{
	auto zeros = cv::Mat::zeros(cv::Size(1920, 454), CV_8UC1);

	double maxVal = 0;
	cv::minMaxLoc(zeros, 0, &maxVal, 0, 0);
}

BOOST_AUTO_TEST_CASE(cv_mat_memory_leak_4)
{
	cv::Mat zeros = cv::Mat::zeros(cv::Size(1920, 454), CV_8UC1);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> contour;
	contour.push_back(cv::Point(2, 2));
	contour.push_back(cv::Point(100, 100));
	contours.push_back(contour);
	cv::drawContours(zeros, contours, 0, cv::Scalar(255, 0, 0), cv::LINE_4, 8);

}

BOOST_AUTO_TEST_CASE(cv_memory_leak_all, *boost::unit_test::disabled())
{
	// Disabled: cv::imshow requires GUI support not available in CI
	cv::Mat zeros = cv::Mat::zeros(cv::Size(1920, 454), CV_8UC1);
	zeros = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);

	double maxVal = 0;
	cv::minMaxLoc(zeros, 0, &maxVal, 0, 0);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> contour;
	contour.push_back(cv::Point(2, 2));
	contour.push_back(cv::Point(100, 100));
	contours.push_back(contour);
	cv::drawContours(zeros, contours, 0, cv::Scalar(255, 0, 0), cv::LINE_4, 8);

	int bins = 16;
	cv::Mat outImg;
	cv::Mat maskImg;
	int channelNumber = 0;
	float* range = new float[2];
	range[0] = 0;
	range[1] = 256;
	const float** ranges = new const float*[1];
	ranges[0] = range;

	cv::calcHist(&zeros, 1, &channelNumber, maskImg, outImg, 1, &bins, ranges, true, false);

	cv::Mat ones = cv::Mat::ones(cv::Size(16, 1), CV_32FC1);
	ones = ones + ones;
	ones = ones / 2;
	cv::compareHist(ones, ones, 1);

	for (auto i = 0; i < 100; i++)
	{
#ifndef LINUX
		cv::imshow("", zeros);
		cv::waitKey(1);
#endif		
	}

	delete[] range;
	delete[] ranges;

}


BOOST_AUTO_TEST_SUITE_END()